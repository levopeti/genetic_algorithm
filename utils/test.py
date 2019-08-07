import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import pickle
import os


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        self.fig = plt.figure()
        self.ax1, self.ax2 = self.fig.subplots(2, 1)
        # self.ax2 = self.fig.add_subplot(2, 1, 1)

        self.line1 = Line2D([], [], color='black', linewidth=2)
        self.line2 = Line2D([], [], color='red', linewidth=2)
        # self.line3 = Line2D([], [], color='green', linewidth=2)

        self.ax1.add_line(self.line1)
        self.ax2.add_line(self.line2)
        # self.ax2.add_line(self.line3)

        animation.TimedAnimation.__init__(self, self.fig, interval=200, blit=True)

    def _draw_frame(self, framedata):
        log_dict, valid_keys, iter_keys = framedata

        key1 = valid_keys[0]
        key2 = valid_keys[1]
        y_data1 = log_dict[key1]
        y_data2 = log_dict[key2]

        iter_key1 = key1.split('/')[0] + "/iteration"
        iter_key2 = key2.split('/')[0] + "/iteration"
        x_data1 = log_dict[iter_key1]
        x_data2 = log_dict[iter_key2]

        # self.ax1.set_title(key1)
        # self.ax2.set_title(key2)

        self.line1.set_data([0, 1, 2], [100, 80, 90])
        # self.line1.set_data(x_data1, y_data1)
        # self.line3.set_data(x_data1, y_data1)
        self.line2.set_data(x_data2, y_data2)

        self._drawn_artists = [self.line1, self.line2]  # , self.line3]

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()

    def new_frame_seq(self):
        while True:
            with open(os.path.join("/home/biot/projects/research/logs", "log"), "rb") as log_file:
                logs = pickle.load(log_file)

            log_dict = dict()
            for log in logs:
                for key, inner_dict in log.items():
                    for inner_key, item in inner_dict.items():
                        if key + '/' + inner_key not in log_dict.keys():
                            log_dict[key + '/' + inner_key] = []
                        log_dict[key + '/' + inner_key] += [item]

            keys = list(log_dict.keys())
            valid_keys = list(filter(lambda key: key.split('/')[1] != "iteration", keys))
            iter_keys = list(filter(lambda key: key.split('/')[1] == "iteration", keys))

            yield log_dict, valid_keys, iter_keys

    def _init_draw(self):
        lines = [self.line1, self.line2]  # , self.line3]
        for l in lines:
            l.set_data([], [])


ani = SubplotAnimation()
# ani.save('test_sub.mp4')
plt.show()
