import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


N = 10_000


def acceptance_rejection(lam, rng):
    """Generate N samples from Gamma(2, 1) by acceptance-rejection."""
    c = 1.0 / (np.e * lam * (1.0 - lam))
    samples = np.empty(N)
    accepted = 0
    proposals = 0

    start_time = perf_counter()
    while accepted < N:
        proposal_u = rng.random()
        y = -np.log(1.0 - proposal_u) / lam

        f_y = y * np.exp(-y)
        g_y = lam * np.exp(-lam * y)
        acceptance_u = rng.random()

        proposals += 1
        if acceptance_u < f_y / (c * g_y):
            samples[accepted] = y
            accepted += 1
    elapsed_time = perf_counter() - start_time

    acceptance_fraction = accepted / proposals
    mean_time = elapsed_time / accepted
    return samples, c, acceptance_fraction, mean_time


def run_experiment(lam, plot_filename, rng):
    samples, c, acceptance_fraction, mean_time = acceptance_rejection(lam, rng)

    print(f"lambda = {lam}")
    print(f"c = {c:.6f}")
    print(f"Empirical acceptance fraction = {acceptance_fraction:.6f}")
    print(f"Theoretical acceptance rate = {1.0 / c:.6f}")
    print(f"Mean computation time per accepted sample = {mean_time:.9e} seconds")
    print()

    x = np.linspace(0.0, max(10.0, samples.max()), 500)
    theoretical_density = x * np.exp(-x)

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.65,
             label="Accepted samples")
    plt.plot(x, theoretical_density, "r-", linewidth=2,
             label=r"$f(x)=xe^{-x}$")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title(f"Acceptance-Rejection Sampling, lambda = {lam}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.close()


rng = np.random.default_rng(2026)
run_experiment(0.5, "HW2/problem3/problem3_lambda_05.png", rng)
run_experiment(0.2, "HW2/problem3/problem3_lambda_02.png", rng)
