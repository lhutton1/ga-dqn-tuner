"""
A series of matplotlib plots used in the project.
"""
import pickle

import numpy as np
from matplotlib import pyplot as plt


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
            print("SAVING MODEL")
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


def comparison_plot(save_path, save_name, title, x_label, y_label, first_data, second_data):
    figure, axes = plt.subplots()
    axes.set_autoscalex_on(True)
    axes.set_autoscaley_on(True)

    figure.suptitle(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)

    scores_stack_1 = np.dstack(tuple(x for x in first_data))[0]
    avg_scores_1 = np.mean(scores_stack_1, axis=1)
    min_scores_1 = np.percentile(scores_stack_1, 10, axis=1)
    max_scores_1 = np.percentile(scores_stack_1, 90, axis=1)
    scores_stack_2 = np.dstack(tuple(x for x in second_data))[0]
    avg_scores_2 = np.mean(scores_stack_2, axis=1)
    min_scores_2 = np.percentile(scores_stack_2, 10, axis=1)
    max_scores_2 = np.percentile(scores_stack_2, 90, axis=1)
    steps = np.arange(0, len(first_data[0]))

    plt.plot(steps, avg_scores_1, '-r', label="ga-dqn")
    plt.fill_between(steps, min_scores_1, max_scores_1, facecolor=(1, 0, 0, .3))
    plt.plot(steps, avg_scores_2, '-g', label="ga")
    plt.fill_between(steps, min_scores_2, max_scores_2, facecolor=(0, 1, 0, .3))
    plt.legend(loc="lower right")
    plt.show()

    figure.savefig(save_path + "/" + save_name + ".png")
