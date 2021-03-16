"""
Tuner with genetic algorithm, using reinforcement learning for crossover and mutation.

DISCLAIMER: Some of the code below is from the GATuner included in the standard TVM package.
            Comments have been added in order to identify where these sections of code have
            been used, although other sections may have some similarities when compared to
            the TVM package.
"""

import logging
from pathlib import Path
import json
from collections import Counter

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


class Transition:
    def __init__(self, prev_state, state, action, gene, score=None):
        self.prev_state = prev_state
        self.state = state
        self.action = action
        self.gene = gene
        self.score = score

        self.input = None
        self.result = None


class DQNGATuner(Tuner):
    def __init__(self,
                 task,
                 learn_start=100,
                 target_update_frequency=50,
                 train_frequency=8,
                 discount=0.99,
                 epsilon_decay=0.99,
                 agent_batch_size=32,
                 memory_capacity=200,
                 debug=True):
        super(DQNGATuner, self).__init__(task)

        # Initialise debugging
        if debug:
            self.initialise_debugging(discount, memory_capacity)

        # Search space info
        self.space = task.config_space
        self.dims = [len(v) for v in self.space.space_map.values()]
        self.visited = set([])
        self.trial_pt = 0

        # GA Tuner
        self.pop_size = min(16, len(self.space))
        self.elite_num = 3
        self.elite_population = []
        self.population = []

        # RL Agent
        self.learn_start = learn_start
        self.update_frequency = target_update_frequency
        self.train_frequency = train_frequency
        self.agent_batch_size = agent_batch_size
        self.epsilon = (1.0, 0.05, epsilon_decay)
        self.mutation_agent, self.crossover_agent = self.create_rl_agents(discount, memory_capacity)

        # RL Training
        self.mutation_prev_fitness = 0
        self.crossover_prev_fitness = 0
        self.step_count = 0

    def create_rl_agents(self, discount, memory_capacity):
        """
        Create DQN agents for both mutation and crossover.
        """
        eps_max, eps_min, eps_decay = self.epsilon
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
        state_space_size = len(self.dims) * 2
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
        import matplotlib as mpl
        mpl.use('Agg')
        from matplotlib import pyplot as plt
        from tools.plots import DynamicPlot, DualDynamicPlot, DynamicScatterPlot
        self.debug = True

        # Monitoring
        plt.ion()
        self.loss_plot = DualDynamicPlot("DQN Mean Squared Error loss", "steps", "loss value", "mutation", "crossover")
        self.avg_score_plot = DynamicPlot("Running average GFLOPS (avg. previous 100 measurements)", "steps",
                                          "GFLOPS")
        self.action_plot_mutate = DynamicScatterPlot("Mutation action selection", "steps",
                                                     "knob mutation index (-1 is no mutation)")
        self.action_plot_crossover = DynamicScatterPlot("Crossover action selection", "steps", "knob crossover index")
        self.best_score_plot = DynamicPlot("Best GFLOPS", "steps", "GFLOPS")
        self.reward_plot = DualDynamicPlot("Reward received", "steps", "DQN reward received", "mutation", "crossover")

        self.memory_capacity = memory_capacity
        self.discount = discount
        self.scores = []

        # Logging
        #ch = logging.StreamHandler(sys.stdout)
        #ch.setLevel(logging.DEBUG)
        #formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
        #ch.setFormatter(formatter)
        #logger.addHandler(ch)
        #logger.setLevel(logging.DEBUG)

    def reserve_elites(self):
        """
        Swap elite genes with elite genes from previous population.
        """
        scores = [t.score for t in self.population]
        elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num:]
        self.elite_population = []
        for idx in elite_indexes:
            self.elite_population.append(self.population[idx])

    def rl_mutate(self):
        """
        Mutate genes using DQN to suggest which knob to randomly mutate.

        DISCLAIMER: This section of code uses the mutation algorithm from GATuner
                    as a base. It has been extended with RL action selection and
                    debugging.
        """
        for i, transition in enumerate(self.population):
            gene = transition.gene
            next_gene = gene.copy()

            if len(self.visited) < len(self.space):
                action = self.mutation_agent.select_action(gene)
                # Action value of 0 means no mutation occurs
                if action != 0:
                    next_gene[action-1] = np.random.randint(self.dims[action-1])

                    # If next gene already visited, fallback to random mutation.
                    while knob2point(next_gene, self.dims) in self.visited:
                        action = np.random.randint(len(self.dims))
                        next_gene[action] = np.random.randint(
                            self.dims[action]
                        )

                self.population[i] = Transition(gene, next_gene, action, next_gene)
                self.visited.add(knob2point(gene, self.dims))
            else:
                break

        # Debugging
        if self.debug:
            occurrences = Counter(t.action for t in self.population)
            for action, occurrence in occurrences.items():
                marker_size = occurrence * 2
                self.action_plot_mutate.update_plot(self.step_count, action-1, marker_size)

    def rl_crossover(self):
        """
        Crossover genes using DQN to suggest the crossover point.

        DISCLAIMER: This section of code uses the mutation algorithm from GATuner
                    as a base. It has been extended with RL action selection and
                    debugging.
        """
        scores = np.array([t.score for t in self.population])
        indices = np.arange(len(self.population))
        scores += 1e-8
        scores /= np.max(scores)
        probabilities = scores / np.sum(scores)
        tmp_genes = []

        for i in range(self.pop_size):
            p1, p2 = np.random.choice(indices, size=2, replace=False, p=probabilities)
            p1, p2 = self.population[p1].gene, self.population[p2].gene
            # TODO the agent should be aware of both the genes involved in cross over. Could the difference be taken?
            #state = p1 + p2
            #point = self.crossover_agent.select_action(state)
            point = np.random.randint(len(self.dims))
            next_gene = p1[:point] + p2[point:]
            #next_state = next_gene + [0] * len(next_gene)
            tmp_genes.append(Transition(None, None, point, next_gene))

        self.population = tmp_genes

        # Debugging
        if self.debug:
            occurrences = Counter(t.action for t in self.population)
            for action, occurrence in occurrences.items():
                marker_size = occurrence * 2
                self.action_plot_crossover.update_plot(self.step_count, action, marker_size)

    def update(self, agent):
        """
        Update DQN agent after receiving results.
        """
        scale = 1e9
        reward_multiplier = 5
        is_crossover = agent.name == "crossover"
        prev_fitness = self.crossover_prev_fitness if is_crossover else self.mutation_prev_fitness

        if self.step_count >= self.pop_size:
            for i, transition in enumerate(self.population):
                # Calculate reward
                score = transition.score
                if score >= self.best_flops:
                    reward = (score * reward_multiplier) / scale
                elif prev_fitness < score <= self.best_flops:
                    reward = score / scale
                elif prev_fitness == score:
                    reward = 0
                else:
                    reward = (score * -1) / scale
                agent.memory.store([transition.prev_state, transition.action, transition.state, reward, False])

                # Train DQN
                steps = self.step_count + i
                if steps > self.learn_start and i % self.train_frequency == 0:
                    loss = agent.train(self.agent_batch_size)
                    agent.update_epsilon()
                    if self.debug and loss is not None:
                        self.loss_plot.update_plot(steps, loss.item(), is_crossover)
                if steps % self.update_frequency == 0:
                    agent.update_target_net()

                if self.debug:
                    self.scores.append(score)
                    if not is_crossover and steps >= 100:
                        average_score = round(np.mean(self.scores[-100:]) / scale, 2)
                        self.avg_score_plot.update_plot(steps, average_score)
                    self.best_score_plot.update_plot(steps, round(np.max(self.best_flops) / scale, 2))
                    self.reward_plot.update_plot(steps, reward, is_crossover)

        no_transitions = len(self.population)
        if is_crossover:
            self.crossover_prev_fitness = np.mean(self.scores[-no_transitions:])
        else:
            self.mutation_prev_fitness = np.mean(self.scores[-no_transitions:])

    def has_next(self):
        """
        Return true to continue tuning, false if not.
        """
        return len(self.visited) < len(self.space)

    def measure_configs(self, n_parallel, measure_batch):
        """
        Measure results for current population.
        """
        # Iterate ceil(no.self.transitions / n_parallel) number of times
        for i in range((len(self.population) + (n_parallel - 1)) // n_parallel):
            configs = []
            batch_size = min(n_parallel, len(self.population) - (i * n_parallel))
            transitions_offset = (i * n_parallel) - 1
            for j in range(transitions_offset, transitions_offset + batch_size):
                gene = self.population[j].gene
                configs.append(self.space.get(knob2point(gene, self.dims)))
            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            # TODO remove end time value. Custom results gives ([result], end time) rather than [result]
            results = measure_batch(inputs)[0]
            for j in range(len(results)):
                transition = self.population[transitions_offset + j]
                input, result = inputs[j], results[j]
                transition.input = inputs[j]
                transition.result = results[j]
                transition.score = input.task.flop / np.mean(result.costs) if result.error_no == 0 else 0.0

    def save_model(self, save_path, save_name):
        """
        Save model to file.
        """
        abs_path = Path(save_path + save_name).resolve()
        abs_path.mkdir(exist_ok=True, parents=True)
        abs_path_str = str(abs_path)
        self.mutation_agent.save_models(abs_path_str + "/mutate_policy_net.model",
                                        abs_path_str + "/mutate_target_net.model")
        self.crossover_agent.save_models(abs_path_str + "/crossover_policy_net.model",
                                         abs_path_str + "/crossover_target_net.model")

        if self.debug:
            from matplotlib import pyplot as plt

            self.loss_plot.save(abs_path_str, "loss")
            self.avg_score_plot.save(abs_path_str, "avg_score")
            self.best_score_plot.save(abs_path_str, "best_score")
            self.action_plot_mutate.save(abs_path_str, "action_mutate")
            self.action_plot_crossover.save(abs_path_str, "action_crossover")
            self.reward_plot.save(abs_path_str, "reward")

            params = {
                "Learn Start": self.learn_start,
                "Update Frequency": self.update_frequency,
                "Discount": self.discount,
                "Agent Batch Size": self.agent_batch_size,
                "Epsilon": list(self.epsilon),
                "Memory Capacity": self.memory_capacity,
                "Dims": len(self.dims),
                "Space Size": len(self.space),
                "Best Score": self.best_flops,
                "Best Config": self.best_config.to_json_dict()
            }
            with open(abs_path_str + "/params.json", "w") as f:
                json.dump(params, f, indent=4, sort_keys=True)

            # close plots
            plt.close(self.loss_plot.figure)
            plt.close(self.avg_score_plot.figure)
            plt.close(self.best_score_plot.figure)
            plt.close(self.action_plot_mutate.figure)
            plt.close(self.action_plot_crossover.figure)
            plt.close(self.reward_plot.figure)

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
        """
        GADQNTuner requires custom tuning pipeline as it requires partial measurement of genes
        after crossover, before mutation.

        DISCLAIMER: In order to customise the tuning pipeline we had to reimplement the tune
                    function. This method is mostly taken from Tuner with the exception of
                    an implementation of a custom tuning pipeline.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, "n_parallel", 1)
        early_stopping = early_stopping or 1e9
        format_si_prefix(0, si_prefix)
        GLOBAL_SCOPE.in_tuning = True

        while self.step_count < n_trial:
            if not self.has_next():
                break

            if self.step_count < self.pop_size:
                # Initialise population. DISCLAIMER: This code is from GATuner.
                for _ in range(self.pop_size):
                    gene = point2knob(np.random.randint(len(self.space)), self.dims)
                    while knob2point(gene, self.dims) in self.visited:
                        gene = point2knob(np.random.randint(len(self.space)), self.dims)
                    transition = Transition(None, None, None, gene)
                    self.population.append(transition)
                    self.visited.add(knob2point(gene, self.dims))

                self.measure_configs(n_parallel, measure_batch)
                self.reserve_elites()
            else:
                self.population.extend(self.elite_population)
                self.reserve_elites()
                self.rl_crossover()
                #self.measure_configs(n_parallel, measure_batch)
                #self.update(self.crossover_agent)
                self.rl_mutate()
                self.measure_configs(n_parallel, measure_batch)
                self.update(self.mutation_agent)

            # Keep best config. DISCLAIMER most of the remaining code is from Tuner.
            for k, transition in enumerate(self.population):
                if transition.score > self.best_flops:
                    self.best_flops = transition.score
                    self.best_config = transition.input.config
                    self.best_measure_pair = (transition.input, transition.result)
                    self.best_iter = self.step_count + k

            self.step_count += len(self.population)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - self.step_count

            for callback in callbacks:
                inputs = [t.input for t in self.population]
                results = [t.result for t in self.population]
                callback(self, inputs, results)

            if self.step_count >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch
