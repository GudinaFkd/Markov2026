import csv
import json
import urllib.request
from pathlib import Path
URL = 'https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?seasontype=2&groups=4&limit=200'
def main():
    rows = []
    for season in (2025, 2026):
        for w in range(1, 16):
            with urllib.request.urlopen(f'{URL}&dates={season}&week={w}', timeout=30) as f:
                d = json.load(f)
            for ev in d.get('events', []):
                c = ev['competitions'][0]
                cp = c['competitors']
                if not all((str(x['team'].get('conferenceId', '')) == '4' for x in cp)):
                    continue
                H = next((x for x in cp if x['homeAway'] == 'home'))
                A = next((x for x in cp if x['homeAway'] == 'away'))
                done = int(bool(c['status']['type']['completed']))
                sc = lambda x: int(float(x['score'])) if done else 'NaN'
                row = (season, w, ev['date'][:10], H['team']['location'], sc(H), A['team']['location'], sc(A), int(bool(c['neutralSite'])), done)
                if row not in rows:
                    rows.append(row)
    output_path = Path(__file__).resolve().parent / 'outputs' / 'big12.csv'
    output_path.parent.mkdir(exist_ok=True)
    with output_path.open('w', newline='') as f:
        out = csv.writer(f)
        out.writerow(['season', 'week', 'date', 'home', 'home_pts', 'away', 'away_pts', 'neutral', 'completed'])
        out.writerows(rows)
    n25 = sum((r[0] == 2025 for r in rows))
    n26 = sum((r[0] == 2026 and r[8] == 1 for r in rows))
    print(f'{len(rows)} games written to {output_path} (2025: {n25}, 2026 played: {n26}, 2026 to play: {len(rows) - n25 - n26})')


if __name__ == '__main__':
    main()
