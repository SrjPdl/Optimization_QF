import numpy as np

class Quadratic:
    def f(self, x):
        return np.arange(1, len(x) + 1) @ (x ** 2)

    def f_p(self, x):
        return 2 * np.arange(1, len(x) + 1).reshape(-1, 1) * x

    def f_pp(self, x):
        return np.diag(2 * np.arange(1, len(x) + 1))
    
    def get_initial_x(self):
        return np.linspace(0, 1, 100).reshape(-1,1)
    
    def __repr__(self):
        return f"{type(self).__name__}"