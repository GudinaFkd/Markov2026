from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def transition(m):
    P = np.zeros((m, m))
    P[np.arange(m-1), np.arange(1, m)] = 1
    P[-1, :2] = .5
    return P


def main():
    out = Path(__file__).resolve().parent / 'outputs'
    out.mkdir(exist_ok=True)
    rows = []
    for m in range(3, 13):
        P = transition(m)
        support = P > 0
        B = np.eye(m, dtype=bool)
        M = 0
        while not B.all():
            B = (B.astype(int) @ support.astype(int)) > 0
            M += 1
        modulus = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        assert M == (m-1)**2 + 1
        rows.append((m, M, (m-1)**2+1, modulus[1]))
    with (out / 'exponents.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['m', 'M', 'Wielandt_bound', 'second_eigenvalue_modulus'])
        w.writerows(rows)
    data = np.array(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(data[:, 0], data[:, 1], 'o', label='Computed M')
    axes[0].plot(data[:, 0], data[:, 2], '--', label='(m-1)^2 + 1')
    axes[0].set_ylabel('Primitive exponent')
    axes[0].legend()
    axes[1].plot(data[:, 0], data[:, 3], 'o-')
    axes[1].set_ylabel(r'$|\lambda_2|$')
    for ax in axes:
        ax.set_xlabel('m')
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'wielandt.png', dpi=160)
    plt.close(fig)
    for m, M, bound, slem in rows:
        print(f'm={m:2d}, M={M:3d}, bound={bound:3d}, |lambda_2|={slem:.9f}')
    P = transition(4)
    pi = np.array([1, 2, 2, 2])/7
    assert np.allclose(pi @ P, pi)
    B = np.eye(4, dtype=int)
    def semigroup(t):
        return any(t-3*a >= 0 and (t-3*a) % 4 == 0
                   for a in range(max(t//3+1, 0)))

    for n in range(41):
        assert bool(B[0, 0]) == (n == 0 or semigroup(n-4))
        assert bool(B[3, 3]) == semigroup(n)
        B = (B @ (P > 0)) > 0


if __name__ == '__main__':
    main()
