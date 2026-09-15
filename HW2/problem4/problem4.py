import numpy as np
import matplotlib.pyplot as plt


# Mixture parameters
a = 0.9
lambda_f = 1000.0
lambda_s = 10.0
N = 100_000

# A fixed seed makes the simulation reproducible.
rng = np.random.default_rng(2026)

# Choose the fast component with probability a and the slow component otherwise.
fast_component = rng.random(N) < a
rates = np.where(fast_component, lambda_f, lambda_s)

# Generate exponential dwell times by inverse transform.
u = rng.random(N)
dwell_times = -np.log(1.0 - u) / rates

empirical_mean = dwell_times.mean()
theoretical_mean = a / lambda_f + (1.0 - a) / lambda_s

cutoff = 0.050
empirical_tail = np.mean(dwell_times > cutoff)
theoretical_tail = (a * np.exp(-lambda_f * cutoff)
                    + (1.0 - a) * np.exp(-lambda_s * cutoff))

print(f"Empirical mean = {empirical_mean:.9f} seconds")
print(f"Theoretical mean = {theoretical_mean:.9f} seconds")
print(f"Empirical P(T > 0.050 s) = {empirical_tail:.6f}")
print(f"Theoretical P(T > 0.050 s) = {theoretical_tail:.6f}")

x = np.linspace(0.0, dwell_times.max(), 1000)
mixture_density = (a * lambda_f * np.exp(-lambda_f * x)
                   + (1.0 - a) * lambda_s * np.exp(-lambda_s * x))

plt.figure(figsize=(8, 5))
plt.hist(dwell_times, bins=200, density=True, alpha=0.65,
         label="Simulated dwell times")
plt.plot(x, mixture_density, "r-", linewidth=2,
         label="Theoretical mixture density")
plt.yscale("log")
plt.xlabel("Dwell time (seconds)")
plt.ylabel("Density (log scale)")
plt.title("Mixture of Fast and Slow Exponential Dwell Times")
plt.legend()
plt.tight_layout()
plt.savefig("HW2/problem4/problem4_dwell_times.png", dpi=150)
plt.close()
