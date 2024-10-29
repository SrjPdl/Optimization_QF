import os
import numpy as np
import matplotlib.pyplot as plt
from functions.log_barrier import LogBarrier

log_barrier = LogBarrier()

def back_tracking_line_search(x, p, f, f_p, alpha=1, c=0.1, rho=0.5, check_domain=False):
    while alpha >= 1e-10:  # Avoid too-small alpha to prevent infinite loop
        new_x = x + alpha * p
        if (check_domain and isin_domain(new_x)) or \
           (not check_domain and f(new_x) <= f(x) + c * alpha * np.dot(p.T, f_p(x))):
            break
        alpha *= rho
    else:
        print("Warning: Alpha became too small.")
    return alpha

def isin_domain(x):
    return np.all(log_barrier.b - np.dot(log_barrier.A, x) > 0)


def plot_error_curve(error, func_name, method_name, curves_path):
    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("||f(x) - f(x-1)||")
    plt.title(f"Error curve for {func_name} function with {method_name}")
    plt.grid(visible=True, which="both", linestyle="--", color="gray", alpha=0.5)
    
    
    plot_path = os.path.join(curves_path, f"{func_name}_{method_name}_error_curve.png")
    plt.savefig(plot_path)
    plt.close()

def plot_convergence_curve(fn_values, func_name, method_name, curves_path):
    plt.plot(fn_values)
    plt.xlabel("Iterations")
    plt.ylabel("f(x)")
    plt.title(f"Convergence curve for {func_name} function with {method_name}")
    plt.grid(visible=True, which="both", linestyle="--", color="gray", alpha=0.5)
    
    
    plot_path = os.path.join(curves_path, f"{func_name}_{method_name}_convergence_curve.png")
    plt.savefig(plot_path)
    plt.close()

def save_results_to_text(x_opt, errors, num_steps, func_name, method_name, results_path):

    file_path = os.path.join(results_path, f"{func_name}_{method_name}_results.txt")
    errors = [float(error) for error in errors]

    with open(file_path, "w") as file:
        file.write(f"Optimization Results for {func_name} using {method_name}\n")
        file.write("-" * 40 + "\n")
        file.write(f"Optimal x: {x_opt}\n")
        file.write(f"Number of steps: {num_steps}\n")
        file.write("Errors: ")
        file.write(str(errors))
