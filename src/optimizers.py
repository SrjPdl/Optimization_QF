
import numpy as np
from utils import back_tracking_line_search

def gradient_descent(x_0, f, f_p, f_pp = None, delta1=1e-3, delta2=1e-3, max_iter=1000, check_domain = False):
  x = x_0
  errors = []
  function_values =[]
  iter = 0
  while True:
      function_values.append(f(x)[0])
      x_old = x
      p = -f_p(x)
      alpha = back_tracking_line_search(x, p, f, f_p, check_domain=check_domain)
      x_new = x + alpha*p
      x = x_new
      errors.append(np.linalg.norm(f(x_new) - f(x_old)))
      iter += 1
      if(np.linalg.norm(f(x_new) - f(x_old)) <= delta1 and np.linalg.norm(x - x_new) <= delta2) or iter > max_iter:
        break

  return x, f(x), errors, function_values

def newton_method(x_0, f, f_p, f_pp, delta1=1e-3, delta2=1e-3, max_iter=1000, check_domain=False):
  x = x_0
  errors = []
  function_values = []
  iter = 0
  while True:
      function_values.append(f(x)[0])
      x_old = x
      p = np.linalg.solve(f_pp(x), -f_p(x))
      alpha = back_tracking_line_search(x, p, f, f_p, check_domain=check_domain)
      x_new = x + alpha*p
      x = x_new
      errors.append(np.linalg.norm(f(x_new) - f(x_old)))
      iter += 1
      if(np.linalg.norm(f(x_new) - f(x_old)) <= delta1 and np.linalg.norm(x - x_new) <= delta2) or iter > max_iter:
        break

  return x, f(x), errors, function_values


def quasi_newton_method(x_0, f, f_p, f_pp, delta1=1e-3, delta2=1e-3, max_iter=1000, check_domain = False):
  errors = []
  function_values = []
  x = x_0
  F = np.eye(x.shape[0])
  iter = 0
  while True:
      function_values.append(f(x)[0])
      x_old = x
      p = F @ -f_p(x)
      alpha = back_tracking_line_search(x, p, f, f_p, check_domain=check_domain)
      x_new = x + alpha*p
      x = x_new
      s = x - x_old
      y = f_p(x) - f_p(x_old)
      second_term = (y.T @ (F@y + s)/np.power((y.T@s), 2))*s@s.T
      third_term = -(s@y.T@F + F@y@s.T)/(y.T@s)
      F = F + second_term + third_term
      iter += 1
      errors.append(np.linalg.norm(f(x_new) - f(x_old)))
      if(np.linalg.norm(f(x_new) - f(x_old)) <= delta1 and np.linalg.norm(x - x_new) <= delta2) or iter > max_iter:
        break

  return x, f(x), errors, function_values


def adam_optimizer(x_0, f, f_p, f_pp, delta1=None, delta2=None, beta1=0.9, beta2=0.999, alpha=0.01, epsilon=1e-8, max_iters=3000, tol=1e-4, check_domain=False):
    x = x_0.astype(float)
    M = np.zeros_like(x, dtype=float)
    V = np.zeros_like(x, dtype=float)
    errors = []
    function_values = []
    for i in range(1, max_iters+1):
        function_values.append(f(x)[0])
        x_old = x.copy()
        grad = f_p(x)

        M = beta1 * M + (1 - beta1) * grad
        V = beta2 * V + (1 - beta2) * (grad ** 2)

        M_hat = M / (1 - beta1 ** i)
        V_hat = V / (1 - beta2 ** i)

        alpha_t = alpha * np.sqrt(1 - beta2 ** i) / (1 - beta1 ** i)
        x -= alpha_t * M_hat / (np.sqrt(V_hat) + epsilon)
        errors.append(np.linalg.norm(f(x) - f(x_old)))

        if errors[-1] < tol:
            break

    return x, f(x), errors, function_values