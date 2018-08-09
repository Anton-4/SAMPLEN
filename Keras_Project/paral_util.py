import queue
from multiprocessing import Queue


class Drain(object):

    def __init__(self, q: Queue):
        self.q = q

    def __iter__(self):
        while True:
            try:
                item = self.q.get_nowait()
                yield item
            except queue.Empty:
                break
