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
            bar = axes.bar(x + x_offset, 
                           y, 
                           width=bar_width * single_width, 
                           color=colors[i % len(colors)])

        bars.append(bar[0])

    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xticks(np.arange(len(results["ticks"])))
    axes.set_xticklabels(results["ticks"])
    axes.legend(bars, [s["name"] for s in results["strategies"]])



"""
A series of predefined results used in the report.
"""

SEARCH_STRATEGY_TUNING_RESULTS = {
    "ticks":    ["Mobilenet v2", "Resnet 18", "Inception v3", "BERT", "Transformer"],
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
            "name": "XGB",
            "data": [277.43417, 204.20383, 0.0, 41.42767, 85.39717]
        }
    ]
}

SEARCH_STRATEGY_BENCHMARK_RESULTS = {
    "ticks":    ["Mobilenet v2", "Resnet 18", "Inception v3", "BERT", "Transformer"],
    "strategies": [
        {
            "name": "Random",
            "data": [0.00282, 0.00133, 0.00857, 0.00778, 0.0039]
        },
        {
            "name": "GA",
            "data": [0.00109, 0.00137, 0.0072, 0.00536, 0.00318]
        },
        {
            "name": "XGB",
            "data": [0.00095, 0.0013, 0, 0.00763, 0.00302]
        },
        {
            "name": "PyTorch",
            "data": [0.00564, 0.00237, 0.01193, 0.00811, 0.00159]
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
                     "Tuning time of different search strategies on different workloads.")
    plt.show()


def get_search_strategy_benchmark_results():
    """
    Quick helper function to get search strategy benchmark results.
    """
    grouped_bar_plot(SEARCH_STRATEGY_BENCHMARK_RESULTS, 
                     "Workload", 
                     "Time (seconds) - Lower is better", 
                     "Execution time of different workloads after being tuned.")
    plt.show()
