from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def outbreak(N, K, R0):
    # Log products and log-sum-exp avoid overflow for subcritical parameters.
    log_rho = np.r_[0., np.cumsum(np.log(N/(R0*(N-np.arange(1, K)))))]
    shift = log_rho.max()
    return np.exp(-shift)/np.exp(log_rho-shift).sum()


def mean_steps(N, K, R0, dt, gamma=1.):
    n = np.arange(1, K)
    p = R0/N * gamma*n*(N-n)*dt
    q = gamma*n*dt
    assert np.all(p+q <= 1)
    A = np.diag(p+q) - np.diag(p[:-1], 1) - np.diag(q[1:], -1)
    e = np.linalg.solve(A, np.ones(K-1))
    assert np.max(np.abs(A @ e - 1)) < 1e-10
    return e


def simulate(N, K, R0, dt, trials=100000, seed=2026):
    rng = np.random.default_rng(seed)
    states = np.ones(trials, dtype=int)
    steps = np.zeros(trials, dtype=int)
    active = np.arange(trials)
    while active.size:
        n = states[active]
        p = R0/N*n*(N-n)*dt
        q = n*dt
        u = rng.random(active.size)
        states[active] += (u < p).astype(int) - (u >= 1-q).astype(int)
        steps[active] += 1
        active = active[(states[active] > 0) & (states[active] < K)]
    times = steps*dt
    return times.mean(), times.std(ddof=1)/np.sqrt(trials)


def main():
    out = Path(__file__).resolve().parent / 'outputs'
    out.mkdir(exist_ok=True)
    grid = np.linspace(.5, 3, 501)
    curves = [np.array([outbreak(N, N//10, r) for r in grid]) for N in (100, 1000)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for N, curve in zip((100, 1000), curves):
        ax.plot(grid, curve, label=f'N={N}, K={N//10}')
    ax.plot(grid, 1-1/grid, '--', label=r'$1-1/R_0$ (unclipped)')
    ax.plot(grid, np.maximum(0, 1-1/grid), ':', color='black', label='Limiting probability')
    ax.set(xlabel=r'$R_0$', ylabel='Probability of reaching K from 1', ylim=(-1.05, .75))
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / 'outbreak_probability.png', dpi=160)
    plt.close(fig)
    np.savetxt(out / 'outbreak_probability.csv', np.column_stack((grid, *curves, 1-1/grid)), delimiter=',', header='R0,h1_N100,h1_N1000,1_minus_1_over_R0', comments='')
    rows = []
    for r in (.5, .8, 1.5, 3.):
        e = mean_steps(100, 10, r, .005)
        assert np.allclose(.005*e, .0025*mean_steps(100, 10, r, .0025))
        rows.append((r, e[0], .005*e[0]))
    with (out / 'hitting_times.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['R0', 'e1_steps', 'e1_time'])
        w.writerows(rows)
    mean, se = simulate(100, 10, 1.5, .005)
    exact = rows[2][2]
    report = '\n'.join(f'R0={r:g}: e1*dt={t:.9f}' for r, _, t in rows)
    report += f'\nSimulation R0=1.5, 100000 paths, seed=2026: mean={mean:.9f}, SE={se:.9f}\n95% normal CI: [{mean-1.96*se:.9f}, {mean+1.96*se:.9f}]; exact={exact:.9f}\n'
    (out / 'results.txt').write_text(report)
    print(report)
    assert abs(mean-exact) < 5*se


if __name__ == '__main__':
    main()
