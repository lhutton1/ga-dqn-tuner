"""
Exact copy of tuner with genetic algorithm from TVM (python/tvm/autotvm/tuner/ga_tuner.py)
with added debugging.

DISCLAIMER: THIS WORK IS NOT MY OWN. It has been added here for convenience and testing purposes.
It is part of the TVM project.
"""

from os import mkdir
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from tvm.autotvm.tuner import Tuner
from tvm.autotvm.tuner.model_based_tuner import knob2point, point2knob

from tools.plots import DynamicPlot, DynamicScatterPlot


class GATuner(Tuner):
    """Tuner with genetic algorithm.
    This tuner does not have a cost model so it always run measurement on real machines.
    This tuner expands the :code:`ConfigEntity` as gene.

    Parameters
    ----------
    pop_size: int
        number of genes in one generation
    elite_num: int
        number of elite to keep
    mutation_prob: float
        probability of mutation of a knob in a gene
    """

    def __init__(self, task, pop_size=100, elite_num=3, mutation_prob=0.1, debug=False):
        super(GATuner, self).__init__(task)

        # algorithm configurations
        self.pop_size = pop_size
        self.elite_num = elite_num
        self.mutation_prob = mutation_prob

        assert elite_num <= pop_size, "The number of elites must be less than population size"

        # space info
        self.space = task.config_space
        self.dim_keys = []
        self.dims = []
        for k, v in self.space.space_map.items():
            self.dim_keys.append(k)
            self.dims.append(len(v))

        self.visited = set([])

        # current generation
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

        self.debug = debug
        if self.debug:
            plt.ion()
            self.best_score_plot = DynamicPlot("Best score", "steps", "best score")
            self.action_plot = DynamicScatterPlot("Action selection", "steps", "action value")

    def next_batch(self, batch_size):
        ret = []
        for _ in range(batch_size):
            gene = self.genes[self.trial_pt % self.pop_size]
            self.trial_pt += 1
            ret.append(self.space.get(knob2point(gene, self.dims)))

        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / np.mean(res.costs)
                self.scores.append(y)
            else:
                self.scores.append(0.0)

        self.steps += len(results)
        if self.debug:
            self.best_score_plot.update_plot(self.steps, round(self.best_flops / 1000000000, 2))

        if len(self.scores) >= len(self.genes) and len(self.visited) < len(self.space):
            genes = self.genes + self.elites
            scores = np.array(self.scores[: len(self.genes)] + self.elite_scores)

            # reserve elite
            self.elites, self.elite_scores = [], []
            elite_indexes = np.argpartition(scores, -self.elite_num)[-self.elite_num :]
            for ind in elite_indexes:
                self.elites.append(genes[ind])
                self.elite_scores.append(scores[ind])

            # cross over
            indices = np.arange(len(genes))
            scores += 1e-8
            scores /= np.max(scores)
            probs = scores / np.sum(scores)
            tmp_genes = []
            for _ in range(self.pop_size):
                p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
                p1, p2 = genes[p1], genes[p2]
                point = np.random.randint(len(self.dims))
                tmp_gene = p1[:point] + p2[point:]
                tmp_genes.append(tmp_gene)

            # mutation
            next_genes = []
            transitions = []
            for tmp_gene in tmp_genes:
                for j, dim in enumerate(self.dims):
                    if np.random.random() < self.mutation_prob:
                        tmp_gene[j] = np.random.randint(dim)

                if len(self.visited) < len(self.space):
                    j = -1
                    while knob2point(tmp_gene, self.dims) in self.visited:
                        j = np.random.randint(len(self.dims))
                        tmp_gene[j] = np.random.randint(
                            self.dims[j]
                        )  # pylint: disable=invalid-sequence-index
                    next_genes.append(tmp_gene)
                    transitions.append(j)
                    self.visited.add(knob2point(tmp_gene, self.dims))
                else:
                    break

            if self.debug:
                occurrences = Counter(transitions)
                for action, occurrence in occurrences.items():
                    marker_size = occurrence * 2
                    self.action_plot.update_plot(self.steps, action, marker_size)

            self.genes = next_genes
            self.trial_pt = 0
            self.scores = []

    def save_model(self, save_path, save_name):
        """Save the current model."""
        if self.debug:
            abs_path = Path(save_path + save_name).resolve()
            abs_path.mkdir(exist_ok=True, parents=True)
            abs_path_str = str(abs_path)
            self.best_score_plot.save(abs_path_str, "best_score")
            self.action_plot.save(abs_path_str, "action")
            plt.close(self.best_score_plot.figure)

    def has_next(self):
        return len(self.visited) - (len(self.genes) - self.trial_pt) < len(self.space)

    def load_history(self, data_set):
        pass
