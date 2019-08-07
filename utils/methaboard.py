import pickle

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.lines import Line2D


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1

    def prev(self, event):
        self.ind -= 1


class MethaBoard(object):

    def __del__(self):
        plt.close()

    @staticmethod
    def reload_dict(file_path):
        while True:
            try:
                with open(file_path, "rb") as log_file:
                    logs = pickle.load(log_file)
            except FileNotFoundError:
                print("Wrong file path!")
                break

            log_dict = dict()
            for log in logs:
                for key, inner_dict in log.items():
                    for inner_key, item in inner_dict.items():
                        if key + '/' + inner_key not in log_dict.keys():
                            log_dict[key + '/' + inner_key] = []
                        log_dict[key + '/' + inner_key] += [item]

            keys = list(log_dict.keys())
            key_begins = list(set([key.split('/')[0] for key in keys]))
            key_begins.sort()

            yield log_dict, key_begins

    def run(self, file_path):
        global animation

        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)

        l1 = Line2D([], [], color='brown', lw=2)
        l2 = Line2D([], [], color='green', lw=2)
        l3 = Line2D([], [], color='blue', lw=2)
        l4 = Line2D([], [], color='black', lw=2)
        l5 = Line2D([], [], color='red', lw=2)

        ax1.add_line(l1)
        ax1.add_line(l2)
        ax2.add_line(l3)
        ax3.add_line(l4)
        ax4.add_line(l5)

        lines = [l1, l2, l3, l4, l5]
        axes = [ax1, ax2, ax3, ax4]

        def update(frame):
            log_dict, key_begins = frame

            best_fitness = log_dict["iteration_end/best_fitness"]
            global_best_fitness = log_dict["iteration_end/global_best_fitness"]
            x_data = log_dict["iteration_end/iteration"]

            l1.set_data(x_data, global_best_fitness)
            l2.set_data(x_data, best_fitness)

            y_data3 = log_dict[key_begins[callback.ind % len(key_begins)] + "/improvement"]
            y_data4 = log_dict[key_begins[callback.ind % len(key_begins)] + "/sum_of_eval"]
            y_data5 = log_dict[key_begins[callback.ind % len(key_begins)] + "/step_time"]
            x_data2 = log_dict[key_begins[callback.ind % len(key_begins)] + "/iteration"]

            l3.set_data(x_data2, y_data3)
            l4.set_data(x_data2, y_data4)
            l5.set_data(x_data2, y_data5)

            ax1.legend(["global_best_fitness", "best_fitness"])
            ax2.legend([key_begins[callback.ind % len(key_begins)] + " improvement"])
            ax3.legend([key_begins[callback.ind % len(key_begins)] + " sum_of_eval"])
            ax4.legend([key_begins[callback.ind % len(key_begins)] + " step_time"])

            for ax in axes:
                ax.relim()
                ax.autoscale_view()

            return lines

        callback = Index()
        animation = FuncAnimation(fig, update, frames=self.reload_dict(file_path), blit=False, interval=800)

        axprev = plt.axes([0.7, 0.01, 0.1, 0.03])
        axnext = plt.axes([0.81, 0.01, 0.1, 0.03])

        bnext = Button(axnext, 'Next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        bprev.on_clicked(callback.prev)

        plt.show()


if __name__ == "__main__":
    mb = MethaBoard()
    mb.run("/home/biot/projects/research/logs/log")
