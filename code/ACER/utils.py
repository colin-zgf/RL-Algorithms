from multiprocessing import Process, Value, Lock


class Counter(object):
    def __init__(self, init=0):
        self.val = Value('i', init)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value
