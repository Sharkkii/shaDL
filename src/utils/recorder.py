# Recorder

class Recorder1d:

    def __init__(self, n_capacity=100, alpha=0.5):
        self.data = []
        self.n = 0
        self.average = None
        self.simple_moving_average = None
        self.exponential_moving_average = None
        self.n_capacity = n_capacity
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        self.data = []
        self.n = 0
        self.average = None
        self.simple_moving_average = None
        self.exponential_moving_average = None
    
    def record(self, x):
        self.data.append(x)
        self.n += 1
        if (self.n == 1):
            self.average = self.simple_moving_average = self.exponential_moving_average = x
        else:
            self.average = ((self.n-1) * self.average + x) / self.n
            self.simple_moving_average = self.average if (self.n <= self.n_capacity) else self.simple_moving_average + (x - self.data[-(self.n_capacity+1)]) / self.n_capacity
            self.exponential_moving_average = (1 - self.alpha) * self.exponential_moving_average + self.alpha * x

