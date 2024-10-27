import numpy as np

class Rosenbrock:

    def f(self, x):
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def f_p(self, x):
        return np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 
                        200 * (x[1] - x[0]**2)])
    
    def f_pp(self, x):
        x = np.asarray(x).flatten()  
        return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], 
                        [-400 * x[0], 200]])
    
    def get_initial_x(self):
        return np.array([[2],[2]])
    
    def __repr__(self):
        return f"{type(self).__name__}"