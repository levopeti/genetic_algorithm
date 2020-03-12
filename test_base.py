# used by test_caller.py
import os
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import gc


class Logger(object):
    def __init__(self):
        print("inited, pid:{}".format(os.getpid()))
        self.counter = 0
        self.pid = os.getpid()

    def increment(self):
        self.counter += 1
        print(self.counter)
        # print(id(self))
        # print(self.pid)
        print(os.getpid())
        print()

    def __del__(self):
        print("deleted, pid:{}".format(os.getpid()))


# logger = Logger()
BaseManager.register('Logger', Logger)
manager = BaseManager()
manager.start()
logger = manager.Logger()


def increment_counter(i):
    print(os.getpid())
    logger.increment()


