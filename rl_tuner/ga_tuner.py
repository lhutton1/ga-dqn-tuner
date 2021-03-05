"""
Tuner with genetic algorithm, using reinforcement learning for crossover and mutation.
"""

import logging
from pathlib import Path
import json
import sys
from collections import Counter
import tempfile

import numpy as np
import torch

from tvm.autotvm.tuner import Tuner
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.util import format_si_prefix
from tvm.autotvm.env import GLOBAL_SCOPE

from .model import DQNAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("autotvm")


class DQNGATuner(Tuner):
    def __init__(self,
                 task,
                 learn_start=100,
                 update_frequency=50,
                 discount=0.99,
                 agent_batch_size=32,
                 memory_capacity=200,
                 debug=False):
        super(DQNGATuner, self).__init__(task)

        # Search space info
        self.space = task.config_space
        self.dims = [len(v) for v in self.space.space_map.values()]
        self.visited = set([])
        self.trial_pt = 0

        # GA Tuner
        self.pop_size = 16
        self.elite_num = 3
        assert self.elite_num < self.pop_size, "The number of elites must be less than population size."

        # RL agent
        self.learn_start = learn_start
        self.update_frequency = update_frequency
        self.agent_batch_size = agent_batch_size
        self.mutation_agent, self.crossover_agent = self.create_rl_agents(discount, memory_capacity)

        # Initialise population
        self.observations = []
        self.pop_size = min(self.pop_size, len(self.space))
        for _ in range(self.pop_size):
            gene = point2knob(np.random.randint(len(self.space)), self.dims)
            while knob2point(gene, self.dims) in self.visited:
                gene = point2knob(np.random.randint(len(self.space)), self.dims)
            self.observations.append((None, None, gene))
            self.visited.add(knob2point(gene, self.dims))

        # RL training
        self.mutation_prev_fitness = 0
        self.crossover_prev_fitness = 0
        self.step_count = 0

        # Initialise debugging
        if debug:
            self.initialise_debugging(discount, memory_capacity)

    def create_rl_agents(self, discount, memory_capacity):
        """
        Create DQN agents for both mutation and crossover.
        """
        eps_max, eps_min, eps_decay = 1.0, 0.05, 0.95
        state_space_size = len(self.dims)
        action_space_size = len(self.dims) + 1
        mutation_agent = DQNAgent("mutate",
                                  device,
                                  state_space_size,
                                  action_space_size,
                                  discount=discount,
                                  eps_max=eps_max,
                                  eps_min=eps_min,
                                  eps_decay=eps_decay,
                                  memory_capacity=memory_capacity)
        state_space_size = len(self.dims)
        action_space_size = len(self.dims) - 1
        crossover_agent = DQNAgent("crossover",
                                   device,
                                   state_space_size,
                                   action_space_size,
                                   discount=discount,
                                   eps_max=eps_max,
                                   eps_min=eps_min,
                                   eps_decay=eps_decay,
                                   memory_capacity=memory_capacity)
        return mutation_agent, crossover_agent

    def initialise_debugging(self, discount, memory_capacity):
        """
        Start the tuner with debugging.
        """
        from matplotlib import pyplot as plt
        from tools.plots import DynamicPlot, DynamicScatterPlot
        self.debug = True

        # Monitoring
        plt.ion()
        self.loss_plot = DynamicPlot("MSE Loss", "steps", "loss")
        self.avg_score_plot = DynamicPlot("Average score", "steps", "average score")
        self.action_plot = DynamicScatterPlot("Mutation action selection", "steps", "action value")
        self.best_score_plot = DynamicPlot("Best score", "steps", "best score")
        self.reward_plot = DynamicPlot("Reward received", "steps", "reward value")
        self.memory_capacity = memory_capacity
        self.discount = discount

        # Logging
        #ch = logging.StreamHandler(sys.stdout)
        #ch.setLevel(logging.DEBUG)
        #formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
        #ch.setFormatter(formatter)
        #logger.addHandler(ch)
        #logger.setLevel(logging.DEBUG)

    def reserve_elites(self, genes, scores):
        """
        Reserve elite genes.
        """
        elites, elite_scores = [], []
        elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num:]
        for ind in elite_indexes:
            elites.append(genes[ind])
            elite_scores.append(scores[ind])
        return elites, elite_scores

    def rl_mutate(self, genes):
        """
        Mutate genes using DQN to suggest which knob to randomly mutate.
        """
        transitions = []
        for gene in genes:
            next_gene = gene.copy()

            if len(self.visited) < len(self.space):
                action = self.mutation_agent.select_action(gene)
                # action value of 0 means no mutation occurs
                if action != 0:
                    next_gene[action-1] = np.random.randint(self.dims[action-1])
                transitions.append((gene, action, next_gene))
                self.visited.add(knob2point(gene, self.dims))
            else:
                break

        # Debugging
        if self.debug:
            occurrences = Counter(t[1] for t in transitions)
            for action, occurrence in occurrences.items():
                marker_size = occurrence * 2
                self.action_plot.update_plot(self.step_count, action-1, marker_size)

        return transitions

    def rl_crossover(self, genes, scores):
        """
        Crossover genes using DQN to suggest the crossover point.
        """
        transitions = []
        indices = np.arange(len(genes))
        scores += 1e-8
        scores /= np.max(scores)
        probabilities = scores / np.sum(scores)

        for _ in range(self.pop_size):
            p1, p2 = np.random.choice(indices, size=2, replace=False, p=probabilities)
            p1, p2 = genes[p1], genes[p2]
            # TODO the agent should be aware of both the genes involved in cross over. Could the difference be taken?
            point = self.crossover_agent.select_action(p1)
            next_gene = p1[:point] + p2[point:]
            transitions.append((p1, point, next_gene))

        return transitions

    def update(self, agent, observations, scores):
        """
        Update DQN agent after receiving results.
        """
        loss = None
        scale = 1000000000
        reward_multiplier = 5

        if self.step_count >= self.pop_size:
            for (state, action, next_state), score in zip(observations, scores):
                prev_fitness = self.mutation_prev_fitness if agent.name == "mutate" else self.crossover_prev_fitness
                if score >= self.best_flops:
                    reward = (score * reward_multiplier) / scale
                elif prev_fitness < score <= self.best_flops:
                    reward = score / scale
                elif prev_fitness == score:
                    reward = 0
                else:
                    reward = (score * -1) / scale
                agent.memory.store([state, action, next_state, reward, False])

        if self.step_count > self.learn_start:
            agent.train(self.agent_batch_size)
            if self.step_count % self.update_frequency == 0:
                agent.update_target_net()
            agent.update_epsilon()

        if agent.name == "mutate":
            self.mutation_prev_fitness = np.mean(scores)
        else:
            self.crossover_prev_fitness = np.mean(scores)

    def has_next(self):
        """
        Return true to continue tuning, false if not.
        """
        return len(self.visited) < len(self.space)

    def measure_configs(self, observations, n_parallel, n_trial, measure_batch):
        """
        Measure results for current population.
        """
        inputs, results = [], []
        observation_idx = 0
        while len(results) < len(observations):
            configs = []
            # TODO prevent going over max trials. e.g. min(n_parallel, n_trial - (self.step_count - observation_idx))
            batch_size = n_parallel
            for _ in range(batch_size):
                gene = observations[observation_idx][2]
                observation_idx += 1
                configs.append(self.space.get(knob2point(gene, self.dims)))
            inputs.extend([MeasureInput(self.task.target, self.task, config) for config in configs])
            # TODO remove end time value. Custom results gives ([result], end time) rather than [result]
            results.extend(measure_batch(inputs)[0])

        return inputs, results

    @staticmethod
    def process_results(inputs, results):
        """
        Get scores.
        """
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                yield y
            else:
                yield 0.0

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
        """
        GADQNTuner requires custom tuning pipeline as it requires partial measurement of genes
        after crossover, before mutation.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        format_si_prefix(0, si_prefix)
        old_level = logger.level

        inputs, results, scores, elites, elite_scores = [], [], [], [], []
        errors, error_ct = [], 0
        GLOBAL_SCOPE.in_tuning = True

        while self.step_count < n_trial:
            if not self.has_next():
                break

            if self.step_count < self.pop_size:
                inputs, results = self.measure_configs(self.observations, n_parallel, n_trial, measure_batch)
                scores = list(self.process_results(inputs, results))
            else:
                genes = [transition[2] for transition in self.observations] + elites
                scores = np.array(scores + elite_scores)

                # Elites.
                elites, elite_scores = self.reserve_elites(genes, scores)

                # Crossover.
                transitions = self.rl_crossover(genes, scores)
                inputs, results = self.measure_configs(transitions, n_parallel, n_trial, measure_batch)
                scores = list(self.process_results(inputs, results))
                self.update(self.crossover_agent, transitions, scores)

                # Mutation.
                transitions = self.rl_mutate(genes)
                inputs, results = self.measure_configs(transitions, n_parallel, n_trial, measure_batch)
                scores = list(self.process_results(inputs, results))
                self.update(self.mutation_agent, transitions, scores)

            # Keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1
                    error = res.costs[0]
                    if isinstance(error, str):
                        errors.append(error)
                    else:
                        errors.append(str(error))

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = self.step_count + k

                logger.debug(
                    "No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
                    self.step_count + k + 1,
                    si_prefix,
                    format_si_prefix(flops, si_prefix),
                    format_si_prefix(self.best_flops, si_prefix),
                    res,
                    config)

            self.step_count += len(scores)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - self.step_count

            for callback in callbacks:
                callback(self, inputs, results)

            if self.step_count >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Switching to debug mode.")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        if error_ct == self.step_count:
            _, f = tempfile.mkstemp(prefix="tvm_tuning_errors_", suffix=".log", text=True)
            with open(f, "w") as file:
                file.write("\n".join(errors))
            logging.warning(
                "Could not find any valid schedule for task %s. "
                "A file containing the errors has been written to %s.",
                self.task,
                f)

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch
