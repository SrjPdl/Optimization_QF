import os
from functions.quadratic import Quadratic
from functions.log_barrier import LogBarrier
from functions.rosenbrock import Rosenbrock

from optimizers import gradient_descent, newton_method, quasi_newton_method, adam_optimizer
from utils import plot_error_curve, plot_convergence_curve, save_results_to_text

def main():
    CURVES_PATH = "./artifacts/curves"
    RESULTS_PATH = "./artifacts/results"
    os.makedirs(CURVES_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    combinations = [
        (Quadratic, gradient_descent),
        (Quadratic, newton_method),
        (Quadratic, quasi_newton_method),
        # Bonus
        (Quadratic, adam_optimizer),


        (LogBarrier, gradient_descent),
        (LogBarrier, newton_method),
        (LogBarrier, quasi_newton_method),
        # Bonus
        (LogBarrier, adam_optimizer),


        (Rosenbrock, gradient_descent),
        (Rosenbrock, newton_method),
        (Rosenbrock, quasi_newton_method),
        # Bonus
        (Rosenbrock, adam_optimizer),

    ]

    for func, method in combinations:
        function = func()
        x_0 = function.get_initial_x()

        if(repr(function) == "LogBarrier"):
            x_opt, f_x, errors, fn_values = method(x_0, function.f, function.f_p, function.f_pp, delta1 = 1e-2, delta2= 1e-2, check_domain=True)
        else:
            x_opt, f_x, errors, fn_values = method(x_0, function.f, function.f_p, function.f_pp, delta1 = 1e-2, delta2= 1e-2, check_domain=False)
        
        plot_error_curve(errors, repr(function), method.__name__, CURVES_PATH)
        plot_convergence_curve(fn_values, repr(function), method.__name__, CURVES_PATH)
        save_results_to_text(x_opt, errors, len(errors) - 1, repr(function), method.__name__, RESULTS_PATH)
        print(f"{repr(function)} function => {method.__name__}: min_fx: {f_x}, steps: {len(errors) - 1}")
        print()


if __name__ == "__main__":
    main()