"""
A series of matplotlib plots used in the project.
"""
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


class DynamicPlot:
    """
    Create a matplotlib graph which can be dynamically updated
    as more points become available.
    """
    def __init__(self, title="", x_label="", y_label="", x_data=None, y_data=None):
        self.figure, self.axes = plt.subplots()
        self.axes.set_autoscalex_on(True)
        self.axes.set_autoscaley_on(True)

        self.title = title
        self.x_label = x_label
        self.y_label = y_label

        self.figure.suptitle(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

        self.x_data = x_data if x_data else []
        self.y_data = y_data if y_data else []
        self.lines, = self.axes.plot(self.x_data, self.y_data)

    def update_plot(self, new_x, new_y):
        self.x_data.append(new_x)
        self.y_data.append(new_y)
        self.lines.set_xdata(self.x_data)
        self.lines.set_ydata(self.y_data)

        self.axes.relim()
        self.axes.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def save(self, save_path, save_name):
        self.figure.savefig(save_path + "/" + save_name + ".png")
        with open(save_path + "/" + save_name + ".pkl", "wb") as f:
            pickle.dump(self.title, f)
            pickle.dump(self.x_label, f)
            pickle.dump(self.y_label, f)
            pickle.dump(self.x_data, f)
            pickle.dump(self.y_data, f)

    @staticmethod
    def load(load_path, load_name):
        with open(load_path + "/" + load_name + ".pkl", "rb") as f:
            title = pickle.load(f)
            x_label = pickle.load(f)
            y_label = pickle.load(f)
            x_data = pickle.load(f)
            y_data = pickle.load(f)

        return DynamicPlot(title, x_label, y_label, x_data, y_data)


class DynamicScatterPlot(DynamicPlot):
    """
    Create a matplotlib graph which can be dynamically updated
    as more points become available.
    """
    def __init__(self, title="",
                 x_label="",
                 y_label="",
                 x_data=None,
                 y_data=None,
                 area=None):
        super(DynamicScatterPlot, self).__init__(title, x_label, y_label, x_data, y_data)
        self.area = area if area else []
        self.scatter = self.axes.scatter(self.x_data, self.y_data, s=self.area)

    def update_plot(self, new_x, new_y, area=1):
        self.x_data.append(new_x)
        self.y_data.append(new_y)
        self.area.append(area)
        self.scatter.set_offsets(np.c_[self.x_data, self.y_data])
        self.scatter.set_sizes(self.area)

        self.axes.ignore_existing_data_limits = True
        self.axes.update_datalim(self.scatter.get_datalim(self.axes.transData))
        self.axes.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def save(self, save_path, save_name):
        self.figure.savefig(save_path + "/" + save_name + ".png")
        with open(save_path + "/" + save_name + ".pkl", "wb") as f:
            pickle.dump(self.title, f)
            pickle.dump(self.x_label, f)
            pickle.dump(self.y_label, f)
            pickle.dump(self.x_data, f)
            pickle.dump(self.y_data, f)
            pickle.dump(self.area, f)

    @staticmethod
    def load(load_path, load_name):
        with open(load_path + "/" + load_name + ".pkl", "rb") as f:
            title = pickle.load(f)
            x_label = pickle.load(f)
            y_label = pickle.load(f)
            x_data = pickle.load(f)
            y_data = pickle.load(f)
            area = pickle.load(f)

        return DynamicScatterPlot(title, x_label, y_label, x_data, y_data, area)


class DualDynamicPlot:
    """
    Create a matplotlib graph which can be dynamically updated
    as more points become available.
    """
    def __init__(self, title="", x_label="", y_label="", data1_label="", data2_label="",
                 x1=None, y1=None, x2=None, y2=None):
        self.figure, self.axes = plt.subplots(1, 1)
        self.axes.set_autoscalex_on(True)
        self.axes.set_autoscaley_on(True)

        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.figure.suptitle(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

        self.data = [
            (x1 if x1 else [], y1 if y1 else []),
            (x2 if x2 else [], y2 if y2 else [])
        ]
        self.lines = []
        self.line_labels = [data1_label, data2_label]
        for i, data in enumerate(self.data):
            self.lines.append(self.axes.plot(*data, label=self.line_labels[i])[0])

    def update_plot(self, new_x, new_y, is_secondary=False):
        idx = 1 if is_secondary else 0
        data = self.data[idx]
        lines = self.lines[idx]
        for i, new_data in enumerate([new_x, new_y]):
            data[i].append(new_data)
        lines.set_xdata(data[0])
        lines.set_ydata(data[1])

        self.axes.relim()
        self.axes.autoscale_view()
        self.axes.legend(loc="lower right")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def save(self, save_path, save_name):
        self.figure.savefig(save_path + "/" + save_name + ".png")
        with open(save_path + "/" + save_name + ".pkl", "wb") as f:
            pickle.dump(self.title, f)
            pickle.dump(self.x_label, f)
            pickle.dump(self.y_label, f)
            pickle.dump(self.line_labels, f)
            pickle.dump(self.data, f)

    @staticmethod
    def load(load_path, load_name):
        with open(load_path + "/" + load_name + ".pkl", "rb") as f:
            title = pickle.load(f)
            x_label = pickle.load(f)
            y_label = pickle.load(f)
            line_labels = pickle.load(f)
            data = pickle.load(f)

        return DualDynamicPlot(title, x_label, y_label, line_labels[0], line_labels[1], data[0][0], data[0][1],
                               data[1][0], data[1][1])


def comparison_plot(save_path, save_name, title, x_label, y_label, y1_data, y2_data, x1_data, x2_data):
    """
    Compares two different tuners (typically ga-dqn and ga for experimentation).
    Takes an average of a number of runs and
    """
    figure, axes = plt.subplots()
    axes.set_autoscalex_on(True)
    axes.set_autoscaley_on(True)

    figure.suptitle(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    scores_stack_1 = np.dstack(tuple(x for x in y1_data))[0]
    avg_scores_1 = np.mean(scores_stack_1, axis=1)
    min_scores_1 = np.percentile(scores_stack_1, 10, axis=1)
    max_scores_1 = np.percentile(scores_stack_1, 90, axis=1)
    scores_stack_2 = np.dstack(tuple(x for x in y2_data))[0]
    avg_scores_2 = np.mean(scores_stack_2, axis=1)
    min_scores_2 = np.percentile(scores_stack_2, 10, axis=1)
    max_scores_2 = np.percentile(scores_stack_2, 90, axis=1)

    plt.plot(x1_data, avg_scores_1, '-r', label="ga-dqn")
    plt.fill_between(x1_data, min_scores_1, max_scores_1, facecolor=(1, 0, 0, .3))
    plt.plot(x2_data[0:], avg_scores_2, '-g', label="ga")
    plt.fill_between(x2_data, min_scores_2, max_scores_2, facecolor=(0, 1, 0, .3))
    plt.legend(loc="lower right")
    plt.show()

    figure.savefig(save_path + "/" + save_name + ".png")


def reward_comparison_plot(save_path, save_name, title, x_label, y_label, y1_data, y2_data, x1_data, x2_data):
    """
    Compares different reward functions against a ga bassline.
    """
    figure, axes = plt.subplots()
    axes.set_autoscalex_on(True)
    axes.set_autoscaley_on(True)

    figure.suptitle(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    reward_avg_scores = []
    reward_min_scores = []
    reward_max_scores = []

    for reward_data in y1_data:
        scores_stack_1 = np.dstack(tuple(x for x in reward_data))[0]
        reward_avg_scores.append(np.mean(scores_stack_1, axis=1))
        reward_min_scores.append(np.percentile(scores_stack_1, 10, axis=1))
        reward_max_scores.append(np.percentile(scores_stack_1, 90, axis=1))

    scores_stack_2 = np.dstack(tuple(x for x in y2_data))[0]
    avg_scores_2 = np.mean(scores_stack_2, axis=1)
    min_scores_2 = np.percentile(scores_stack_2, 10, axis=1)
    max_scores_2 = np.percentile(scores_stack_2, 90, axis=1)

    for reward_idx in range(len(y1_data)):
        p = plt.plot(x1_data, reward_avg_scores[reward_idx], label=f"ga-dqn-R{reward_idx+1}")
        p_colour = p[0].get_color()
        colour_alpha = colors.to_rgba_array(p_colour, alpha=0.3)
        plt.fill_between(x1_data, reward_min_scores[reward_idx], reward_max_scores[reward_idx],
                         facecolor=colour_alpha)

    p = plt.plot(x2_data[0:], avg_scores_2, label="ga")
    p_colour = p[0].get_color()
    colour_alpha = colors.to_rgba_array(p_colour, alpha=0.3)
    plt.fill_between(x2_data, min_scores_2, max_scores_2, facecolor=colour_alpha)
    plt.legend(loc="lower right")
    plt.show()

    figure.savefig(save_path + "/" + save_name + ".png")


def grouped_bar_plot(results, xlabel, ylabel, title):
    """
    A grouped bar plot for comparing different search strategies.
    """
    figure, axes = plt.subplots()
    axes.set_autoscalex_on(True)
    axes.set_autoscaley_on(True)

    n_bars = len(results["strategies"])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    total_width = 0.8
    single_width = 1
    bar_width = total_width / n_bars

    bars = []

    for i, strategy in enumerate(results["strategies"]):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(strategy["data"]):
            if isinstance(y, tuple):
                error = np.array([[y[0] - y[1]], [y[2] - y[0]]])
                bar = axes.bar(x + x_offset,
                               y[0],
                               yerr=error,
                               width=bar_width * single_width,
                               color=colors[i % len(colors)],
                               ecolor='black', capsize=2)
            else:
                bar = axes.bar(x + x_offset,
                               y,
                               width=bar_width * single_width,
                               color=colors[i % len(colors)])

        bars.append(bar[0])

    # hack to add subtitle
    figure.suptitle(title, fontsize=10)
    axes.set_title(subtitle, fontsize=8)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xticks(np.arange(len(results["ticks"])))
    axes.set_xticklabels(results["ticks"])
    axes.legend(bars, [s["name"] for s in results["strategies"]])


def hyper_param_bar_plot(results, xlabel, ylabel, title, limits):
    """
    A bar plot for comparing hyperparameter results.
    """
    figure, axes = plt.subplots()

    axes.set_autoscalex_on(True)
    axes.set_autoscaley_on(True)

    n_bars = len(results["data"])
    bar_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    total_width = 0.8
    single_width = 1
    bar_width = total_width / n_bars

    bars = []

    for x, y in enumerate(results["data"]):
        color = 'red' if x == len(results["data"]) - 1 else bar_color
        error = np.array([[y[0] - y[1]], [y[2] - y[0]]])
        bar = axes.bar(x,
                       y[0],
                       yerr=error,
                       color=color,
                       ecolor='black', capsize=2)
        bars.append(bar[0])

    axes.set_ylim(*limits)

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xticks(np.arange(len(results["ticks"])))
    axes.set_xticklabels(results["ticks"])



"""
A series of predefined results used in the report.

All values are obtained from experiments using the
evaluation framework on ARC3 GPGPU node.
"""

SEARCH_STRATEGY_TUNING_RESULTS = {
    "ticks": ["Mobilenet v2", "Resnet 18", "Inception v3", "BERT", "Transformer"],
    "strategies": [
        {
            "name": "Random",
            "data": [205.36633, 145.87233, 445.783, 23.747, 92.92367]
        },
        {
            "name": "GA",
            "data": [289.41617, 163.81917, 548.72333, 45.44617, 145.0955]
        },
        {
            "name": "GA-DQN",
            "data": [321.35831, 201.28831, 613.25113, 54.3993, 157.30796]
        },
        {
            "name": "XGB",
            "data": [277.43417, 204.20383, 0.0, 41.42767, 85.39717]
        }
    ]
}

SEARCH_STRATEGY_BENCHMARK_RESULTS = {
    "ticks": ["Mobilenet v2", "Resnet 18", "Inception v3", "BERT", "Transformer"],
    "strategies": [
        {
            "name": "Random",
            "data": [(2.82, 2.76, 2.84), (1.33, 1.31, 1.36), (8.57, 8.54, 8.61), (7.78, 7.64, 7.87), (3.9, 3.81, 4.01)]
        },
        {
            "name": "GA",
            "data": [(1.09, 1.04, 1.12), (1.37, 1.32, 1.42), (7.2, 7.13, 7.37), (5.36, 5.21, 5.69), (3.18, 3.13, 3.21)]
        },
        {
            "name": "GA-DQN",
            "data": [(1.15, 1.12, 1.17), (1.23, 1.21, 1.25), (6.85, 6.78, 6.89), (0, 0, 0), (0, 0, 0)]
        },
        {
            "name": "XGB",
            "data": [(0.95, 0.91, 0.98), (1.3, 1.28, 1.35), (0, 0, 0), (7.63, 7.47, 7.71), (3.02, 3.0, 3.05)]
        },
        {
            "name": "PyTorch",
            "data": [(5.64, 5.62, 5.65), (2.37, 2.34, 2.39), (11.93, 11.85, 12.04), (8.11, 7.65, 8.43), (1.59, 1.49, 1.63)]
        }
    ]
}

HYPER_PARAMETER_TRIALS_C1 = {
    "ticks": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "GA"],
    "data": [(4.016, 3.834, 4.234), (4.230, 4.040, 4.413), (4.159, 3.976, 4.282), (4.406, 4.303, 4.592), (4.706, 4.572, 4.804),
             (3.974, 3.725, 4.172), (4.322, 4.125, 4.498), (4.300, 4.168, 4.447), (4.794, 4.703, 4.863), (4.253, 4.215, 4.302),
             (4.683, 4.635, 4.717), (4.609, 4.369, 4.794), (4.254, 4.123, 4.337)]
}

HYPER_PARAMETER_TRIALS_MM1 = {
    "ticks": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "GA"],
    "data": [(361.49, 357.32, 369.24), (383.61, 372.75, 389.38), (368.52, 354.55, 371.32), (363.38, 353.18, 370.27), (383.52, 375.19, 389.46),
             (365.43, 355.39, 369.21), (353.26, 343.32, 361.42), (362.42, 357.29, 368.03), (391.57, 384.72, 395.31), (360.20, 353.54, 365.94),
             (389.26, 382.59, 393.08), (388.32, 379.17, 393.92), (359.01, 351.32, 366.02)]
}

SINGLE_OPERATOR_RESULTS = {
    "ticks": ["C1", "C2", "MM1", "MM2"],
    "strategies": [
        {
            "name": "GA",
            "data": []
        },
        {
            "name": "GA-DQN",
            "data": []
        }
    ]
}


def get_search_strategy_tuning_results():
    """
    Quick helper function to get search strategy tuning results.
    """
    grouped_bar_plot(SEARCH_STRATEGY_TUNING_RESULTS, 
                     "Workload", 
                     "Time (mins) - Lower is better", 
                     "Tuning time of different search strategies on different workloads.",
                     "Completed on ARC3 GPGPU node with tuning settings, early stopping: 250, repeat: 5, trials: 1000")
    plt.show()


def get_search_strategy_benchmark_results():
    """
    Quick helper function to get search strategy benchmark results.
    """
    grouped_bar_plot(SEARCH_STRATEGY_BENCHMARK_RESULTS, 
                     "Workload", 
                     "Time (milliseconds) - Lower is better",
                     "Execution time of different workloads after being tuned.",
                     "Completed on ARC3 GPGPU node after with tuning settings, early stopping: 250, repeat: 5, trials: 1000")
    plt.show()

def get_hyperparameter_results_c1():
    """
    Quick helper function to get hyperparameter results on C1.
    """
    hyper_param_bar_plot(HYPER_PARAMETER_TRIALS_C1,
                         "Hyper-parameter configuration",
                         "Best GigaFLOPS - Higher is better",
                         "Computational performance of C1 after being tuned using GA-DQN with\nvarying hyper-parameters.",
                         (3.5, 5))
    plt.show()

def get_hyperparameter_results_mm1():
    """
    Quick helper function to get hyperparameter results on MM1.
    """
    hyper_param_bar_plot(HYPER_PARAMETER_TRIALS_MM1,
                         "Hyper-parameter configuration",
                         "Best GigaFLOPS - Higher is better",
                         "Computational performance of MM1 after being tuned using GA-DQN with\nvarying hyper-parameters.",
                         (320, 400))
    plt.show()
