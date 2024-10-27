import numpy as np

class LogBarrier:
    def __init__(self) -> None:
        self.b = np.loadtxt('./data/fun2_b.txt').reshape(-1,1)
        self.c = np.loadtxt('./data/fun2_c.txt').reshape(-1,1)
        self.m = self.b.shape[0]
        self.n = self.c.shape[0]
        self.A = np.loadtxt('./data/fun2_A.txt').reshape(self.m, self.n)

    def f(self, X):
        return self.c.T@X - np.sum(np.log(self.b - self.A@X))

    def f_p(self, X):
            return self.c + self.A.T @ (1 / (self.b - self.A@X))

    def f_pp(self, X):
        return self.A.T @ np.diag((1 / np.power((self.b - self.A@X), 2)).flatten()) @ self.A

    def get_initial_x(self):
        return np.ones((self.n,1)) * 0.1

    def __repr__(self):
            return f"{type(self).__name__}"