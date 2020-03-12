import time
import os
import functools
import datetime
import random
import plotly.figure_factory as ff
import plotly.graph_objects as go
from termcolor import colored

from multiprocessing.managers import BaseManager

r = lambda: random.randint(0, 255)


class FunctionTimeBlock(object):
    __slots__ = "start_time", "end_time", "elapsed_time", "name", "level_id", "parent", "pid"

    def __init__(self, name, pid):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

        self.name = name
        self.level_id = None
        self.parent = None
        self.pid = pid

    def __repr__(self):
        return "name: {}, level id: {}, pid: {}".format(self.name, self.level_id, self.pid)

    def start(self):
        self.start_time = datetime.datetime.now()

    def stop(self):
        self.end_time = datetime.datetime.now()
        self.elapsed_time = (self.end_time - self.start_time).total_seconds()

    def set_parent(self, parent):
        self.level_id = parent.level_id + 1
        self.parent = parent


class Profiler(object):
    def __init__(self):
        self.profile_dict = dict()

        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

        self.level_id = 0
        self.pid = os.getpid()
        self.main_pid = None

        self.current_func = {self.pid: self}
        self.keys = list()
        self.func_id = 0

        self.print_pool = False

        self.df = None
        self.summary = [[], [], [], []]
        self.colors = {"Profiler pid: {}".format(self.pid): '#%02X%02X%02X' % (r(), r(), r())}

    def start(self, main_pid):
        self.start_time = datetime.datetime.now()
        self.main_pid = main_pid

    def add_and_start(self, func_name, func_pid):
        if self.main_pid is None:
            print(colored("Profiler isn't active!", color="red"))

        ftb = FunctionTimeBlock(func_name, func_pid)
        ftb.set_parent(self.current_func.get(func_pid, self))

        key = self.make_key(func_pid, ftb.level_id, func_name)
        self.profile_dict[key] = ftb
        self.current_func[func_pid] = ftb
        ftb.start()

        return key

    def end_of_ftb(self, key):
        ftb = self.profile_dict[key]
        ftb.stop()
        self.current_func[ftb.pid] = ftb.parent

    def make_key(self, func_pid, level_id, name):
        new_key = "".join(["_" + str(part) for part in [func_pid, level_id, name, self.func_id]])

        self.keys.append(new_key)
        self.func_id += 1

        return new_key

    def make_df(self):
        self.df = [dict(Task="level {}".format(self.level_id),
                        Start=str(self.start_time),
                        Finish=datetime.datetime.now(),
                        Resource="Profiler pid: {}".format(self.pid))
                   ]
        for key in self.keys:
            ftb = self.profile_dict[key]

            if ftb.end_time is None:
                # in case of multithread
                continue

            if self.print_pool or ftb.pid == self.main_pid:
                resource = "{} pid: {}".format(ftb.name, ftb.pid)
            else:
                resource = "{} other pid".format(ftb.name)

            ftb_dict = dict(Task="level {}".format(ftb.level_id),
                            Start="{}".format(ftb.start_time),
                            Finish="{}".format(ftb.end_time),
                            Resource=resource)

            self.df.append(ftb_dict)
            self.colors[resource] = '#%02X%02X%02X' % (r(), r(), r())

    def make_summary(self):
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        full_time_dict = {0: elapsed_time}
        func_time_dict = dict()

        for key in self.keys:
            ftb = self.profile_dict[key]

            if ftb.pid != self.main_pid:
                continue

            if ftb.end_time is None:
                # in case of multithread
                continue

            full_time_dict[ftb.level_id] = full_time_dict.get(ftb.level_id, 0) + ftb.elapsed_time

            if ftb.level_id not in func_time_dict.keys():
                func_time_dict[ftb.level_id] = dict()

            func_time_dict[ftb.level_id][ftb.name] = func_time_dict[ftb.level_id].get(ftb.name, 0) + ftb.elapsed_time

        for level_id in full_time_dict.keys():
            if level_id == 0:
                # skip profiler (level 0)
                continue

            for name, func_time in func_time_dict[level_id].items():
                # Name
                self.summary[0].append(name)

                # Level
                self.summary[1].append(level_id)

                # Time
                self.summary[2].append("{:04.2f}".format(func_time))

                # Ratio
                self.summary[3].append("{:04.2f}%".format((func_time / full_time_dict[level_id - 1]) * 100))

    def save(self):
        pass

    def print(self):
        fig = ff.create_gantt(self.df, colors=self.colors, index_col='Resource', show_colorbar=True, group_tasks=True,
                              title="Profiler")

        table = go.Table(header=dict(values=['Name', 'Level', 'Time', 'Ratio']),
                         cells=dict(values=self.summary))

        fig.add_traces([table])

        fig['layout']['xaxis2'] = {}
        fig['layout']['yaxis2'] = {}

        fig.layout.yaxis.update({'domain': [0, .4]})
        fig.layout.yaxis2.update({'domain': [.4, 1]})

        fig.layout.update({'height': 400 + len(self.summary[0]) * 40})
        fig.show()

    def __del__(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time


BaseManager.register('Profiler', Profiler)
manager = BaseManager()
manager.start()
profiler = manager.Profiler()

active = False


def profiler_add(func):
    """Add a function to the profiler"""
    @functools.wraps(func)
    def wrapper_profiler(*args, **kwargs):
        global active
        if active:
            key = profiler.add_and_start(func.__name__, os.getpid())

        value = func(*args, **kwargs)
        if active:
            profiler.end_of_ftb(key)

        return value

    return wrapper_profiler


def start_profiler(main_pid):
    global active
    profiler.start(main_pid)
    active = True


def print_profiler():
    global active
    if active:
        profiler.make_df()
        profiler.make_summary()
        profiler.print()


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.6f} secs")
        return value

    return wrapper_timer


if __name__ == "__main__":
    from multiprocessing.pool import Pool

    start_profiler(os.getpid())


    @profiler_add
    def func_1(i=0):
        time.sleep(1)
        func_2()
        return


    @profiler_add
    def func_2(i=0):
        time.sleep(1.5)
        return


    func_1()
    func_2()
    func_1()

    p = Pool(18)
    p.map(func_2, [_ for _ in range(18)])
    print_profiler()
