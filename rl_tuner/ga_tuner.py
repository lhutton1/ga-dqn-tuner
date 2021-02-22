"""
Tuner with genetic algorithm, using reinforcement learning for crossover and mutation.
"""

import logging
from pathlib import Path
import json
import sys
from collections import Counter

import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from tvm.autotvm.tuner import Tuner
from tvm.autotvm.model_based_tuner import knob2point, point2knob

from .model import DQNAgent
from ..tools.plots import DynamicPlot, DynamicScatterPlot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("DQNTuner")


class DQNGATuner(Tuner):
    def __init__(self,
                 task,
                 learn_start=10,
                 update_frequency=50,
                 discount=0.8,
                 agent_batch_size=32,
                 epsilon=(1.0, 0.01, 0.9),
                 memory_capacity=1000,
                 debug=False,
                 pop_size=16,
                 elite_num=3,
                 mutation_prob=0.1):
        super(DQNGATuner, self).__init__(task)

        # space info
        self.space = task.config_space
        self.dim_keys = []
        self.dims = []
        for k, v in self.space.space_map.items():
            self.dim_keys.append(k)
            self.dims.append(len(v))

        self.visited = set([])

        seed = 12345
        np.random.seed(seed)
        torch.manual_seed(seed)

        state_space_size = len(self.dims)
        action_space_size = len(self.dims) + 1
        self.agent = DQNAgent(device,
                              state_space_size,
                              action_space_size,
                              discount=discount,
                              eps_max=epsilon[0],
                              eps_min=epsilon[1],
                              eps_decay=epsilon[2],
                              memory_capacity=memory_capacity)

        self.learn_start = learn_start
        self.update_frequency = update_frequency
        self.discount = discount
        self.agent_batch_size = agent_batch_size

        self.pop_size = pop_size
        self.elite_num = elite_num
        self.mutation_prob = mutation_prob
        assert elite_num <= pop_size, "The number of elites must be less than population size"

        self.genes = []
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.trial_pt = 0
        self.steps = 0
        # random initialization
        self.pop_size = min(self.pop_size, len(self.space))
        self.elite_num = min(self.pop_size, self.elite_num)
        for _ in range(self.pop_size):
            tmp_gene = point2knob(np.random.randint(len(self.space)), self.dims)
            while knob2point(tmp_gene, self.dims) in self.visited:
                tmp_gene = point2knob(np.random.randint(len(self.space)), self.dims)

            self.genes.append(tmp_gene)
            self.visited.add(knob2point(tmp_gene, self.dims))
        self.prev_fitness = 0

        self.observations = [(None, None, gene) for gene in self.genes]
        self.step_count = 0
        self.scores = []
        self.bassline_score = 0

        self.debug = debug
        if self.debug:
            plt.ion()
            self.loss_plot = DynamicPlot("MSE Loss", "steps", "loss")
            self.avg_score_plot = DynamicPlot("Average score", "steps", "average score")
            self.action_plot = DynamicScatterPlot("Mutation action selection", "steps", "action value")
            self.best_score_plot = DynamicPlot("Best score", "steps", "best score")
            self.reward_plot = DynamicPlot("Reward received", "steps", "reward value")
            self.epsilon = epsilon
            self.memory_capacity = memory_capacity
            self.init_debug()

    def rl_mutate(self, tmp_genes):
        mutate_transitions = []
        for tmp_gene in tmp_genes:
            next_gene = tmp_gene.copy()

            if len(self.visited) < len(self.space):
                # TODO skip already visited configs
                #while knob2point(next_gene, self.dims) in self.visited:
                # minus one so negative value represents no mutation
                action = self.agent.select_action(tmp_gene)
                if action != 0:
                    next_gene[action-1] = np.random.randint(
                        self.dims[action-1]
                    )
                mutate_transitions.append((tmp_gene, action, next_gene))
                self.visited.add(knob2point(tmp_gene, self.dims))
            else:
                break

        if self.debug:
            occurrences = Counter(t[1] for t in mutate_transitions)
            for action, occurrence in occurrences.items():
                marker_size = occurrence * 2
                self.action_plot.update_plot(self.step_count, action-1, marker_size)

        return mutate_transitions

    def rl_crossover(self, genes, scores):
        crossover_transitions = []
        tmp_genes = []

        indices = np.arange(len(genes))
        scores += 1e-8
        scores /= np.max(scores)
        probs = scores / np.sum(scores)

        for _ in range(self.pop_size):
            p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
            p1, p2 = genes[p1], genes[p2]
            # TODO choose point with RL.
            point = np.random.randint(len(self.dims))
            tmp_gene = p1[:point] + p2[point:]
            tmp_genes.append(tmp_gene)

        return crossover_transitions, tmp_genes

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            gene = self.observations[self.trial_pt % self.pop_size][2]
            self.trial_pt += 1
            ret.append(self.space.get(knob2point(gene, self.dims)))
        return ret

    def ga_algorithm(self, transitions):
        """
        Standard genetic algorithm with reinforcement learning applied to crossover and mutation.
        """
        genes = [transition[2] for transition in transitions] + self.elites
        scores = np.array(self.scores[: len(genes)] + self.elite_scores)

        # Reserve elite
        self.elites, self.elite_scores = [], []
        elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num:]
        for ind in elite_indexes:
            self.elites.append(genes[ind])
            self.elite_scores.append(scores[ind])

        # Crossover
        cross_transitions, tmp_genes = self.rl_crossover(genes, scores)

        # Mutation
        mutate_transitions = self.rl_mutate(tmp_genes)
        return mutate_transitions

    def update(self, inputs, results):
        loss = None
        scale = 1000000000

        # Collect measurements
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                self.scores.append(y)
            else:
                self.scores.append(0.0)

        # Save a bassline score of initial population. This will be used when calculating reward.
        #if not self.bassline_score:
        #    self.bassline_score = max(self.scores)

        # After each set of measurements, store observations and train DQN
        start_idx = self.trial_pt - len(results)
        end_idx = self.trial_pt
        latest_observations = self.observations[start_idx:end_idx]
        latest_scores = self.scores[start_idx:end_idx]
        if self.step_count >= self.pop_size:
            for (state, action, next_state), score in zip(latest_observations, latest_scores):
                print("SCORE", score, "BEST", self.best_flops, "PREV FITNESS", self.prev_fitness)
                if score >= self.best_flops:
                    reward = (score * 5) / scale
                elif self.prev_fitness < score <= self.best_flops:
                    reward = score / scale
                elif self.prev_fitness == score:
                    reward = 0
                else:
                    reward = (score * -1) / scale
                print("REWARD", reward)
                #reward = (score - self.bassline_score) / scale
                #print("STATE:", state, "ACTION:", action, "NEXTSTATE:", next_state, "INITIAL SCORE:", self.bassline_score / scale, "REWARD (Speedup):", reward, "CURRENT SCORE:", score / scale)
                self.agent.memory.store([state, action, next_state, reward, False])
                if self.debug:
                    self.reward_plot.update_plot(self.step_count, reward)

        if self.step_count > self.learn_start:
            loss = self.agent.train(self.agent_batch_size)
            if self.step_count % self.update_frequency == 0:
                self.agent.update_target_net()
                # Reset bassline score when target updates. This forces better reward.
               # self.bassline_score = 0
            self.agent.update_epsilon()

        self.step_count += len(results)
        self.prev_fitness = np.mean(latest_scores)

        if self.debug:
            score = round(np.max(self.scores) / scale, 2)
            average_score = round(np.mean(self.scores[-100:]) / scale, 2)
            # logger.debug(f"Iteration: {self.step_count}, "
            #              f"Steps: {self.step_count}, "
            #              f"Score: {score}, "
            #              f"Average Score: {average_score}, "
            #              f"Epsilon: {self.agent.epsilon}, "
            #              f"MSE Loss: {loss}")

            if loss is not None:
                self.loss_plot.update_plot(self.step_count, loss.item())
            self.avg_score_plot.update_plot(self.step_count, average_score)
            self.best_score_plot.update_plot(self.step_count, round(self.best_flops / scale, 2))

        # once finished measurements and DQN training, move to next generation and use DQN to create new genes
        if len(self.scores) >= len(self.observations) and len(self.visited) < len(self.space):
            transitions = self.ga_algorithm(self.observations)
            if transitions:
                self.observations = transitions
            self.trial_pt = 0
            self.scores = []

    def save_model(self, save_path, save_name):
        """Save the current model."""
        abs_path = Path(save_path + save_name).resolve()
        abs_path.mkdir(exist_ok=True, parents=True)
        abs_path_str = str(abs_path)
        self.agent.save_models(abs_path_str + "/policy_net.model",
                               abs_path_str + "/target_net.model")
        if self.debug:
            self.loss_plot.figure.savefig(abs_path_str + "/loss.png")
            self.avg_score_plot.figure.savefig(abs_path_str + "/avg_score.png")
            self.best_score_plot.figure.savefig(abs_path_str + "/best_score.png")
            self.action_plot.figure.savefig(abs_path_str + "/action.png")
            self.reward_plot.figure.savefig(abs_path_str + "/reward.png")
            params = {
                "Learn Start": self.learn_start,
                "Update Frequency": self.update_frequency,
                "Discount": self.discount,
                "Agent Batch Size": self.agent_batch_size,
                "Epsilon": self.epsilon,
                "Memory Capacity": self.memory_capacity,
                "Dims": len(self.dims),
                "Space Size": len(self.space),
                "Best Score": self.best_flops,
                "Bassline Score": self.bassline_score,
                "Best Config": self.best_config.to_json_dict()
            }
            with open(abs_path_str + "/params.json", "w") as f:
                json.dump(params, f, indent=4, sort_keys=True)

            # close plots
            plt.close(self.loss_plot.figure)
            plt.close(self.avg_score_plot.figure)
            plt.close(self.best_score_plot.figure)
            plt.close(self.action_plot.figure)
            plt.close(self.reward_plot.figure)

    def has_next(self):
        return len(self.visited) - (len(self.genes) - self.trial_pt) < len(self.space)

    def load_history(self, data_set):
        pass

    @staticmethod
    def init_debug():
        """Debug mode displays more information relating to the RL model being used."""
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
