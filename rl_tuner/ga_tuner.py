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
from matplotlib import pyplot as plt

from tvm.autotvm.tuner import Tuner
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob
from tvm.autotvm.measure import MeasureInput, create_measure_batch
from tvm.autotvm.util import format_si_prefix
from tvm.autotvm.env import GLOBAL_SCOPE

from .model import DQNAgent
from tools.plots import DynamicPlot, DynamicScatterPlot

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
                action = self.agent.select_action(tmp_gene)
                # action value of 0 means no mutation occurs
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
        # if not self.bassline_score:
        #     self.bassline_score = max(self.scores)

        # After each set of measurements, store observations and train DQN
        start_idx = self.trial_pt - len(results)
        end_idx = self.trial_pt
        latest_observations = self.observations[start_idx:end_idx]
        latest_scores = self.scores[start_idx:end_idx]
        reward_multiplier = 5
        if self.step_count >= self.pop_size:
            for (state, action, next_state), score in zip(latest_observations, latest_scores):
                if score >= self.best_flops:
                    reward = (score * reward_multiplier) / scale
                elif self.prev_fitness < score <= self.best_flops:
                    reward = score / scale
                elif self.prev_fitness == score:
                    reward = 0
                else:
                    reward = (score * -1) / scale
                # print("REWARD", reward)
                #reward = (score - self.bassline_score) / scale
                # if score >= self.best_flops:
                #     reward *= 5

                self.agent.memory.store([state, action, next_state, reward, False])
                if self.debug:
                    self.reward_plot.update_plot(self.step_count, reward)

        if self.step_count > self.learn_start:
            loss = self.agent.train(self.agent_batch_size)
            if self.step_count % self.update_frequency == 0:
                self.agent.update_target_net()
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

    # def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
    #     """Begin tuning
    #     Parameters
    #     ----------
    #     n_trial: int
    #         Maximum number of configs to try (measure on real hardware)
    #     measure_option: dict
    #         The options for how to measure generated code.
    #         You should use the return value ot autotvm.measure_option for this argument.
    #     early_stopping: int, optional
    #         Early stop the tuning when not finding better configs in this number of trials
    #     callbacks: List of callable
    #         A list of callback functions. The signature of callback function is
    #         (Tuner, List of MeasureInput, List of MeasureResult)
    #         with no return value. These callback functions will be called on
    #         every measurement pair. See autotvm/tuner/callback.py for some examples.
    #     si_prefix: str
    #         One of tvm.autotvm.utils.SI_PREFIXES. The SI prefix to use when reporting FLOPS.
    #     """
    #     measure_batch = create_measure_batch(self.task, measure_option)
    #     n_parallel = getattr(measure_batch, "n_parallel", 1)
    #     early_stopping = early_stopping or 1e9
    #     self.n_trial = n_trial
    #     self.early_stopping = early_stopping
    #
    #     # Validate si_prefix arg
    #     format_si_prefix(0, si_prefix)
    #
    #     old_level = logger.level
    #
    #     GLOBAL_SCOPE.in_tuning = True
    #     i = error_ct = 0
    #     errors = []
    #     while i < n_trial:
    #         if not self.has_next():
    #             break
    #
    #         # TODO tuner needs a custom pipeline to evaluate crossover reward, then mutation reward.
    #         configs = self.next_batch(min(n_parallel, n_trial - i))
    #
    #         inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
    #         results = measure_batch(inputs)
    #
    #         # keep best config
    #         for k, (inp, res) in enumerate(zip(inputs, results)):
    #             config = inp.config
    #             if res.error_no == 0:
    #                 flops = inp.task.flop / np.mean(res.costs)
    #                 error_ct = 0
    #             else:
    #                 flops = 0
    #                 error_ct += 1
    #                 error = res.costs[0]
    #                 if isinstance(error, str):
    #                     errors.append(error)
    #                 else:
    #                     errors.append(str(error))
    #
    #             if flops > self.best_flops:
    #                 self.best_flops = flops
    #                 self.best_config = config
    #                 self.best_measure_pair = (inp, res)
    #                 self.best_iter = i + k
    #
    #             logger.debug(
    #                 "No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
    #                 i + k + 1,
    #                 si_prefix,
    #                 format_si_prefix(flops, si_prefix),
    #                 format_si_prefix(self.best_flops, si_prefix),
    #                 res,
    #                 config,
    #                 )
    #
    #         i += len(results)
    #         self.ttl = min(early_stopping + self.best_iter, n_trial) - i
    #
    #         self.update(inputs, results)
    #         for callback in callbacks:
    #             callback(self, inputs, results)
    #
    #         if i >= self.best_iter + early_stopping:
    #             logger.debug("Early stopped. Best iter: %d.", self.best_iter)
    #             break
    #
    #         if error_ct > 150:
    #             logging.basicConfig()
    #             logger.warning("Too many errors happen in the tuning. Switching to debug mode.")
    #             logger.setLevel(logging.DEBUG)
    #         else:
    #             logger.setLevel(old_level)
    #
    #     if error_ct == i:
    #         _, f = tempfile.mkstemp(prefix="tvm_tuning_errors_", suffix=".log", text=True)
    #         with open(f, "w") as file:
    #             file.write("\n".join(errors))
    #         logging.warning(
    #             "Could not find any valid schedule for task %s. "
    #             "A file containing the errors has been written to %s.",
    #             self.task,
    #             f,
    #         )
    #     GLOBAL_SCOPE.in_tuning = False
    #     del measure_batch

    def save_model(self, save_path, save_name):
        """Save the current model."""
        abs_path = Path(save_path + save_name).resolve()
        abs_path.mkdir(exist_ok=True, parents=True)
        abs_path_str = str(abs_path)
        self.agent.save_models(abs_path_str + "/policy_net.model",
                               abs_path_str + "/target_net.model")
        if self.debug:
            self.loss_plot.save(abs_path_str, "loss")
            self.avg_score_plot.save(abs_path_str, "avg_score")
            self.best_score_plot.save(abs_path_str, "best_score")
            self.action_plot.save(abs_path_str, "action")
            self.reward_plot.save(abs_path_str, "reward")

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
