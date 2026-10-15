import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = Path(__file__).resolve().parent / 'outputs'

def read_games(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))
def main():
    old = read_games(out.parent / 'data' / 'big12_before_weekend.csv')
    rows = read_games(out / 'big12.csv')
    out.mkdir(exist_ok=True)
    teams = sorted({r[k] for r in rows for k in ('home', 'away')})
    idx = {t: i for i, t in enumerate(teams)}
    team_count = len(teams)

    def games(data, year, cutoff):
        return [r for r in data
                if r['season'] == str(year) and r['completed'] == '1'
                and r['date'] < cutoff]

    def stats(data):
        H = np.zeros((team_count, team_count))
        N = np.zeros((team_count, team_count))
        for g in data:
            i, j = (idx[g['home']], idx[g['away']])
            s, t = (float(g['home_pts']), float(g['away_pts']))
            x = (s + 1) / (s + t + 2)
            h = 0.5 + 0.5 * np.sign(x - 0.5) * np.sqrt(abs(2 * x - 1))
            H[i, j] += h
            H[j, i] += 1 - h
            N[i, j] += 1
            N[j, i] += 1
        return H, N
    flat = np.full((team_count, team_count), 0.5)
    np.fill_diagonal(flat, 0)
    H25, N25 = stats(games(old, 2025, '2026-01-01'))
    last = np.divide(H25, N25, out=flat.copy(), where=N25 > 0)
    s = (H25.sum(axis=1) + 1) / (N25.sum(axis=1) + 2)
    strength_prior = s[:, None] / (s[:, None] + s[None, :])
    np.fill_diagonal(strength_prior, 0)
    priors = {'Flat': flat, 'Last season': last, 'Smoothed 2025 strength': strength_prior}

    def rank(data, prior):
        H, N = stats(data)
        A = (H + prior) / (N.sum(axis=1)[:, None] + team_count - 1)
        vals, vec = np.linalg.eig(A)
        k = np.argmax(vals.real)
        lam = float(vals[k].real)
        r = vec[:, k].real
        r /= r.sum()
        ratio = float(max(abs(np.delete(vals, k))) / lam)
        return lam, ratio, r
    train = games(old, 2026, '2026-10-10')
    updated = games(rows, 2026, '2026-10-14')
    target = sorted(
        (g for g in rows
         if g['season'] == '2026' and g['completed'] == '1'
         and '2026-10-10' <= g['date'] <= '2026-10-11'),
        key=lambda g: (g['date'], g['home']),
    )
    pre = {name: rank(train, p) for name, p in priors.items()}
    post = {name: rank(updated, p) for name, p in priors.items()}
    if not target:
        raise ValueError('No completed games in the evaluation weekend.')
    results = []
    for g in target:
        winner = g['home'] if int(g['home_pts']) > int(g['away_pts']) else g['away']
        entry = {'date': g['date'], 'matchup': g['away'] + ' at ' + g['home'], 'score': g['away_pts'] + '-' + g['home_pts'], 'winner': winner}
        for name, (_, _, r) in pre.items():
            pick = g['home'] if r[idx[g['home']]] >= r[idx[g['away']]] else g['away']
            entry[name] = pick
            entry[name + ' correct'] = int(pick == winner)
        results.append(entry)
    with open(out / 'predictions_evaluated.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    with open(out / 'rankings.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['period', 'prior', 'team', 'strength', 'lambda_star', 'eigenvalue_ratio'])
        for period, models in [('Before Oct 10', pre), ('Before Oct 14', post)]:
            for name, (lam, ratio, r) in models.items():
                for t, v in zip(teams, r):
                    w.writerow([period, name, t, v, lam, ratio])
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharex=True)
    order = np.argsort(pre['Flat'][2])
    for ax, (name, (_, _, r)) in zip(axes, pre.items()):
        ax.barh(np.array(teams)[order], r[order], color='#326b94')
        ax.set_title(name)
        ax.set_xlabel('Normalized strength r')
        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('Big 12 rankings before October 10, 2026 | w = 1')
    fig.tight_layout()
    fig.savefig(out / 'rankings_before_weekend.png', dpi=160)
    plt.close(fig)
    for name in priors:
        correct = sum((e[name + ' correct'] for e in results))
        lam, ratio, _ = pre[name]
        print(f'{name}: {correct}/{len(results)}, lambda={lam:.6f}, ratio={ratio:.6f}')


if __name__ == '__main__':
    main()
