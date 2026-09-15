import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(2026)

# Part (b): inverse-transform sampling from h(z) = 2(1-z), 0 < z < 1
sample_size = 100_000
u = rng.random(sample_size)
z_inverse = 1.0 - np.sqrt(1.0 - u)

empirical_mean_b = z_inverse.mean()
theoretical_mean = 1.0 / 3.0

print("Part (b)")
print(f"Empirical mean = {empirical_mean_b:.6f}")
print(f"Theoretical mean = {theoretical_mean:.6f}")
print()

x = np.linspace(0.0, 1.0, 500)
h_x = 2.0 * (1.0 - x)

plt.figure(figsize=(8, 5))
plt.hist(z_inverse, bins=50, density=True, alpha=0.65,
         label="Inverse-transform samples")
plt.plot(x, h_x, "r-", linewidth=2, label=r"$h(z)=2(1-z)$")
plt.xlabel("z")
plt.ylabel("Density")
plt.title("Problem 5(b): Inverse-Transform Samples")
plt.legend()
plt.tight_layout()
plt.savefig("HW2/problem5/problem5_part_b.png", dpi=150)
plt.close()


# Part (c): grow R core realizations simultaneously from n = 3 to n = N.
R = 1_000
N = 10_000
cores = np.ones(R, dtype=int)  # C(3) = 1 for every realization

for n in range(3, N):
    cores += rng.random(R) < cores / n

z_growth = cores / N
empirical_mean_c = z_growth.mean()
empirical_ratio = z_growth.std() / empirical_mean_c
theoretical_variance = 1.0 / 18.0
theoretical_ratio = np.sqrt(theoretical_variance) / theoretical_mean

print("Part (c)")
print(f"Empirical mean = {empirical_mean_c:.6f}")
print(f"Theoretical mean = {theoretical_mean:.6f}")
print(f"Empirical SD/mean ratio = {empirical_ratio:.6f}")
print(f"Theoretical SD/mean ratio = {theoretical_ratio:.6f}")
print(f"Smallest observed core C = {cores.min()}")
print(f"Largest observed core C = {cores.max()}")

plt.figure(figsize=(8, 5))
plt.hist(z_growth, bins=30, density=True, alpha=0.65,
         label="Core-growth realizations")
plt.plot(x, h_x, "r-", linewidth=2, label=r"$h(z)=2(1-z)$")
plt.xlabel("z = C/N")
plt.ylabel("Density")
plt.title("Problem 5(c): Core Growth Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("HW2/problem5/problem5_part_c.png", dpi=150)
plt.close()
