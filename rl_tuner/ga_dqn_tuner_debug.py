"""
Tuner with genetic algorithm, using reinforcement learning for crossover and mutation.

Additional debugging is included which can be used to track the progress of the
tuner.

DISCLAIMER: Some of the code below is from the GATuner included in the standard TVM package.
            Comments have been added in order to identify where these sections of code have
            been used, although other sections may have some similarities when compared to
            the TVM package.
"""

import logging
from pathlib import Path
import json
from collections import Counter
import enum
import sys
from math import ceil

import numpy as np
# import torch

# from tvm.autotvm.tuner import Tuner
# from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
# from tvm.autotvm.measure import MeasureInput, create_measure_batch
# from tvm.autotvm.util import format_si_prefix
# from tvm.autotvm.env import GLOBAL_SCOPE

# from .dqn_model import DQNAgent

# Debugging packages
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from tools.plots import DynamicPlot, DualDynamicPlot, DynamicScatterPlot


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger = logging.getLogger("autotvm")


class Transition:
    """
    A transition (otherwise called experience) is an object
    that contains data required for each interaction of the agent
    with the environment.
    """
    def __init__(self, prev_state, state, action, gene, score=None):
        self.prev_state = prev_state
        self.state = state
        self.action = action
        self.gene = gene
        self.score = score

        self.input = None
        self.result = None


class RewardFunction(enum.IntEnum):
    """
    Choose between 3 different reward functions.

    R1 - calculate score based on normalised initial score.
    R2 - calculate score based on best flops increase.
    R3 - calculate score using somewhat of a combination of R1 and R2.
    """
    R1 = 0
    R2 = 1
    R3 = 2


class GADQNTuner(Tuner):
    """
    The GA-DQN tuner.

    Applies a genetic algorithm using reinforcement learning
    for mutation and crossover.
    """
    def __init__(self,
                 task,
                 learn_start=100,
                 target_update_frequency=200,
                 train_frequency=4,
                 discount=0.99,
                 epsilon_decay=0.99,
                 agent_batch_size=32,
                 hidden_sizes=(512, 128),
                 learning_rate=5e-3,
                 reward_function=RewardFunction.R3):
        super(GADQNTuner, self).__init__(task)

        # Initialise debugging
        self.initialise_debugging()

        # Search space info
        self.space = task.config_space
        self.dims = [len(v) for v in self.space.space_map.values()]
        self.visited = set([])
        self.trial_pt = 0

        # GA Tuner
        self.pop_size = min(64, len(self.space))
        self.elite_num = 3
        self.elite_population = []
        self.population = []

        # RL Agent
        self.learn_start = learn_start
        self.update_frequency = target_update_frequency
        self.train_frequency = train_frequency
        self.agent_batch_size = agent_batch_size
        self.epsilon = (1.0, 0.1, epsilon_decay)
        self.reward_function = reward_function
        self.discount = discount
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.mutation_agent, self.crossover_agent = None, None  # initialised when tuning begins

        # RL Training
        self.prev_fitness = 0
        self.step_count = 0
        self.mutation_step_count = 0
        self.crossover_step_count = 0
        self.initial_score = 0

    def create_rl_agents(self, discount, memory_capacity, hidden_sizes, learning_rate):
        """
        Create DQN agents for both mutation and crossover.
        """
        state_space_size = len(self.dims)
        action_space_size = len(self.dims) + 1
        mutation_agent = DQNAgent("mutate",
                                  device,
                                  state_space_size,
                                  action_space_size,
                                  discount=discount,
                                  eps=self.epsilon,
                                  memory_capacity=memory_capacity,
                                  hidden_sizes=hidden_sizes,
                                  learning_rate=learning_rate)
        state_space_size = len(self.dims) * 2
        action_space_size = len(self.dims) - 2
        crossover_agent = DQNAgent("crossover",
                                   device,
                                   state_space_size,
                                   action_space_size,
                                   discount=discount,
                                   eps=self.epsilon,
                                   memory_capacity=memory_capacity,
                                   hidden_sizes=hidden_sizes,
                                   learning_rate=learning_rate)
        return mutation_agent, crossover_agent

    def initialise_debugging(self):
        """
        Start the tuner with debugging.
        """
        # Monitoring plots
        plt.ion()
        self.loss_plot = DualDynamicPlot("DQN Mean Squared Error loss",
                                         "steps", "loss value", "mutation", "crossover")
        self.avg_score_plot = DynamicPlot("Running average GFLOPS (avg. previous 100 measurements)",
                                          "steps", "GFLOPS")
        self.action_plot_mutate = DynamicScatterPlot("Mutation action selection", "steps",
                                                     "knob mutation index (-1 is no mutation)")
        self.action_plot_crossover = DynamicScatterPlot("Crossover action selection",
                                                        "steps", "knob crossover index")
        self.best_score_plot = DynamicPlot("Best GFLOPS", "steps", "GFLOPS")
        self.reward_plot = DualDynamicPlot("Reward received", "steps", "DQN reward received",
                                           "mutation", "crossover")

        # Additional tracking
        self.scores = []

    def has_next(self):
        """
        Return true to continue tuning, false if not.
        """
        return len(self.visited) < len(self.space)

    def reserve_elites(self):
        """
        Swap elite genes with elite genes from previous population.
        """
        scores = [t.score for t in self.population]
        elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num:]
        self.elite_population = []
        for idx in elite_indexes:
            self.elite_population.append(self.population[idx])

    def normalise_state(self, state, pad=False):
        """
        Normalise a state to within 0-1 range. This improves training as it
        removes bias from larger values.
        """
        if not pad and len(state) == len(self.dims):
            return np.divide(state, self.dims)
        if len(state) == len(self.dims):
            normalised = np.divide(state, self.dims)
            return np.pad(normalised, (0, len(state)), 'constant', constant_values=0)
        dims = self.dims + self.dims
        return np.divide(state, dims)

    def calculate_reward(self, score):
        """
        Calculate reward based on reward function chosen.
        """
        scale = 1e9
        reward_multiplier = 5  # multiplier used for R3

        if self.reward_function == RewardFunction.R1:
            return (self.initial_score - score) / scale
        elif self.reward_function == RewardFunction.R2:
            reward = score if score >= self.best_flops else 0
            return reward / scale
        elif self.reward_function == RewardFunction.R3:
            if score >= self.best_flops:
                reward = (score * reward_multiplier)
            elif self.prev_fitness < score <= self.best_flops:
                reward = score
            elif self.prev_fitness == score:
                reward = 0
            else:
                reward = (score * -1)

            return reward / scale

    def rl_mutate(self, transitions):
        """
        Mutate genes using DQN to suggest which knob to randomly mutate.
        Mutations happen inplace on the "transitions" that are input.
        """
        for i, transition in enumerate(transitions):
            gene = transition.gene
            next_gene = gene.copy()

            if len(self.visited) < len(self.space):
                action = self.mutation_agent.get_action(gene)
                # An action value of 0 means no mutation occurs
                if action != 0:
                    next_gene[action-1] = np.random.randint(self.dims[action-1])

                # If next gene already visited, fallback to random mutation.
                while knob2point(next_gene, self.dims) in self.visited:
                    action = np.random.randint(len(self.dims))
                    next_gene[action] = np.random.randint(
                        self.dims[action])

                transitions[i] = Transition(self.normalise_state(gene),
                                            self.normalise_state(next_gene),
                                            action, next_gene)
                self.visited.add(knob2point(gene, self.dims))
            else:
                break

        # Debugging
        occurrences = Counter(t.action for t in transitions)
        for action, occurrence in occurrences.items():
            marker_size = occurrence * 2
            self.action_plot_mutate.update_plot(self.mutation_step_count, action-1, marker_size)

    def rl_crossover(self, probabilities, indices, batch_size):
        """
        Crossover genes using DQN to suggest the crossover point.
        """
        tmp_genes = []

        for i in range(batch_size):
            p1, p2 = np.random.choice(indices, size=2, replace=False, p=probabilities)
            p1, p2 = self.population[p1].gene, self.population[p2].gene
            state = p1 + p2
            point = self.crossover_agent.get_action(state)
            next_gene = p1[:point] + p2[point:]
            tmp_genes.append(Transition(self.normalise_state(state),
                                        self.normalise_state(next_gene, pad=True),
                                        point,
                                        next_gene))

        # Debugging
        occurrences = Counter(t.action for t in tmp_genes)
        for action, occurrence in occurrences.items():
            marker_size = occurrence * 2
            self.action_plot_crossover.update_plot(self.crossover_step_count, action, marker_size)

        return tmp_genes

    def mutate_update(self, n_parallel, measure_batch, callbacks):
        """
        Perform RL mutation on the population.
        """
        if self.step_count >= self.pop_size:

            # Batch population by train frequency
            for i in range(ceil(self.pop_size / self.train_frequency)):
                batch_size = min(self.train_frequency, self.pop_size - (i * self.train_frequency))
                transitions_offset = (i * self.train_frequency) - 1

                transitions = self.population[transitions_offset:transitions_offset + batch_size]
                self.rl_mutate(transitions)
                self.measure_configs(transitions, n_parallel, measure_batch, callbacks)

                for j, transition in enumerate(transitions):
                    # Calculate the reward received by the agent.
                    reward = self.calculate_reward(transition.score)

                    self.mutation_agent.memory.store_experience([transition.prev_state,
                                                                 transition.action,
                                                                 transition.state,
                                                                 reward])
                    self.population[transitions_offset + j] = transition

                    if self.mutation_step_count > 0 and \
                            (self.mutation_step_count + j) % self.update_frequency == 0:
                        self.mutation_agent.increment_target()
                    steps = self.step_count - len(transitions) + j
                    self.reward_plot.update_plot(steps, reward, False)

                # Delay the learn start for mutate, as it comes after crossover.
                self.mutation_step_count += batch_size
                if self.mutation_step_count > self.learn_start:
                    loss = self.mutation_agent.train(self.agent_batch_size)
                    self.mutation_agent.reduce_epsilon()
                    if loss is not None:
                        self.loss_plot.update_plot(self.mutation_step_count, loss.item(), False)

        self.prev_fitness = np.mean(self.scores[-self.pop_size:])

    def crossover_update(self, n_parallel, measure_batch, callbacks):
        """
        Perform RL crossover on the population.
        """
        # Calculate crossover probabilities based on whole population
        scores = np.array([t.score for t in self.population])
        scores += 1e-8
        scores /= np.max(scores)
        indices = np.arange(len(self.population))
        probabilities = scores / np.sum(scores)

        if self.step_count >= self.pop_size:
            # Batch population by train frequency
            for i in range(ceil(self.pop_size / self.train_frequency)):
                batch_size = min(self.train_frequency, self.pop_size - (i * self.train_frequency))
                population_offset = (i * self.train_frequency) - 1

                transitions = self.rl_crossover(probabilities, indices, batch_size)
                self.measure_configs(transitions, n_parallel, measure_batch, callbacks)

                for j, transition in enumerate(transitions):
                    # Calculate the reward received by the agent.
                    reward = self.calculate_reward(transition.score)

                    self.crossover_agent.memory.store_experience([transition.prev_state,
                                                                  transition.action,
                                                                  transition.state,
                                                                  reward])
                    self.population[population_offset + j] = transition

                    if self.crossover_step_count > 0 and \
                            (self.crossover_step_count + j) % self.update_frequency == 0:
                        self.crossover_agent.increment_target()
                    steps = self.step_count - len(transitions) + j
                    self.reward_plot.update_plot(steps, reward, True)

                # Train DQN
                self.crossover_step_count += batch_size
                if self.crossover_step_count > self.learn_start:
                    loss = self.crossover_agent.train(self.agent_batch_size)
                    self.crossover_agent.reduce_epsilon()
                    if loss is not None:
                        self.loss_plot.update_plot(self.crossover_step_count, loss.item(), True)

        self.prev_fitness = np.mean(self.scores[-self.pop_size:])

    def measure_configs(self, transitions, n_parallel, measure_batch, callbacks):
        """
        Measure results for current population.
        """
        for i in range(ceil(len(transitions) / n_parallel)):
            configs = []
            batch_size = min(n_parallel, len(transitions) - (i * n_parallel))
            transitions_offset = (i * n_parallel) - 1

            # Get configs
            for j in range(transitions_offset, transitions_offset + batch_size):
                gene = transitions[j].gene
                configs.append(self.space.get(knob2point(gene, self.dims)))

            # Measure batch
            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results, end_time = measure_batch(inputs)

            # Unpack result
            for j in range(len(results)):
                self.step_count += 1
                transition = transitions[transitions_offset + j]
                input, result = inputs[j], results[j]
                transition.input = inputs[j]
                transition.result = results[j]
                transition.score = input.task.flop / np.mean(result.costs) if result.error_no == 0 else 0.0
                self.scores.append(transition.score)

                # Update best
                if transition.score > self.best_flops:
                    self.best_flops = transition.score
                    self.best_config = transition.input.config
                    self.best_measure_pair = (transition.input, transition.result)
                    self.best_iter = self.step_count

                self.best_score_plot.update_plot(self.step_count, round(np.max(self.best_flops) / 1e9, 2))
                if self.step_count >= 100:
                    average_score = round(np.mean(self.scores[-100:]) / 1e9, 2)
                    self.avg_score_plot.update_plot(self.step_count, average_score)

        for callback in callbacks:
            inputs = [t.input for t in transitions]
            results = [t.result for t in transitions]
            callback(self, inputs, results)

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

        self.loss_plot.save(abs_path_str, "loss")
        self.avg_score_plot.save(abs_path_str, "avg_score")
        self.best_score_plot.save(abs_path_str, "best_score")
        self.action_plot_mutate.save(abs_path_str, "action_mutate")
        self.action_plot_crossover.save(abs_path_str, "action_crossover")
        self.reward_plot.save(abs_path_str, "reward")

        params = {
            "Learn Start": self.learn_start,
            "Update Frequency": self.update_frequency,
            "Train Frequency": self.train_frequency,
            "Discount": self.discount,
            "Epsilon": list(self.epsilon),
            "Agent Batch Size": self.agent_batch_size,
            "Hidden Sizes": list(self.hidden_sizes),
            "Learning Rate": self.learning_rate,
            "Reward Function": self.reward_function,
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
        do_crossover = True

        self.mutation_agent, self.crossover_agent = self.create_rl_agents(
            self.discount, int(ceil(n_trial / 2)), self.hidden_sizes, self.learning_rate)

        while self.step_count < n_trial:
            if not self.has_next():
                break

            # Initialise a random population.
            if self.step_count < self.pop_size:
                for _ in range(self.pop_size):
                    gene = point2knob(np.random.randint(len(self.space)), self.dims)
                    while knob2point(gene, self.dims) in self.visited:
                        gene = point2knob(np.random.randint(len(self.space)), self.dims)
                    transition = Transition(None, None, None, gene)
                    self.population.append(transition)
                    self.visited.add(knob2point(gene, self.dims))
                self.measure_configs(self.population, n_parallel, measure_batch, callbacks)
                self.initial_score = np.mean([p.score for p in self.population])
                self.reserve_elites()

            # Apply GA-DQN tuning once initial population has been created.
            else:
                if do_crossover:
                    self.population.extend(self.elite_population)
                    self.reserve_elites()
                    self.crossover_update(n_parallel, measure_batch, callbacks)
                    do_crossover = False
                else:
                    self.mutate_update(n_parallel, measure_batch, callbacks)
                    do_crossover = True

            self.ttl = min(early_stopping + self.best_iter, n_trial) - self.step_count

            if self.step_count >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch
