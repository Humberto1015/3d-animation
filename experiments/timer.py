import time

class Timer:
    def __init__(self):
        self.tStart = None
        self.tEnd = None

        self.time = None

    def start(self):
        self.tStart = time.time()

    def end(self):
        self.tEnd = time.time()

        self.time = self.tEnd - self.tStart