"""
A series of matplotlib plots used in the project.
"""

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


class DynamicScatterPlot:
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
        self.figure, self.axes = plt.subplots()
        self.axes.set_autoscalex_on(True)
        self.axes.set_autoscaley_on(True)

        self.figure.suptitle(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

        self.x_data = x_data if x_data else []
        self.y_data = y_data if y_data else []
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
