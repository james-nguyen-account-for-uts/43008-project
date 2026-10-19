"""Dependency-free, standalone HTML reports for recorded training/evaluation."""
import html
from pathlib import Path


def chart(values, label, color="#38bdf8", successes=None):
  if not values:
    return '<p class="muted">No recorded data available.</p>'
  low, high = min(0, min(values)), max(1, max(values))
  span = high - low
  baseline = 175 - (0 - low) / span * 150
  bars = []
  for index, value in enumerate(values):
    x = 45 + index * 900 / len(values)
    y = 175 - (value - low) / span * 150
    fill = color if successes is None or successes[index] else "#fb7185"
    bars.append(f'<rect x="{x:.2f}" y="{min(y, baseline):.2f}" '
                f'width="{max(0.5, 900 / len(values) - 2):.2f}" '
                f'height="{max(1, abs(y - baseline)):.2f}" fill="{fill}">'
                f'<title>{html.escape(label)} {index + 1}: {value:.2f}</title></rect>')
  return (f'<svg viewBox="0 0 970 210" role="img" aria-label="{html.escape(label)}">'
          f'<text x="2" y="25">{high:.0f}</text><text x="2" y="179">{low:.0f}</text>'
          f'<line x1="45" x2="945" y1="{baseline}" y2="{baseline}" stroke="#64748b"/>'
          + ''.join(bars) + f'<text x="45" y="203">1</text>'
          f'<text x="890" y="203">{len(values)}</text></svg>')


def write_evaluation_report(data, destination):
  """Render actual recorded metrics; never infer missing learning curves."""
  records = data.get("evaluation_maps", [])
  if not records:
    raise ValueError("The report requires per-map evaluation results.")
  summary = data["evaluation"]
  count = len(records)
  success = [row["success_rate"] == 1 for row in records]
  solved = sum(success)
  steps_solved = [row["average_steps"] for row, ok in zip(records, success) if ok]
  solved_average = f'{sum(steps_solved) / len(steps_solved):.1f}' if steps_solved else "N/A"
  history = data.get("training_history", {})
  episode_success = history.get("success", [])
  if episode_success:
    groups = [episode_success[i:i + 100] for i in range(0, len(episode_success), 100)]
    curve = chart([100 * sum(group) / len(group) for group in groups], "Training block")
    training_section = ('<h2>Learning progress</h2><p class="muted">Training success per '
                        '100-episode block; the final block may be shorter. Training includes exploration.</p>'
                        + curve)
  else:
    training_section = ('<h2>Learning progress</h2><p class="muted">This saved run did not record '
                        'episode history. Its final evaluation is shown above. Future training runs '
                        'will record a learning curve automatically.</p>')
  rows = ''.join(f'<tr><td>{i + 1}</td><td>{row["layout_seed"]}</td>'
                 f'<td class="{"good" if ok else "bad"}">{"Solved" if ok else "Failed"}</td>'
                 f'<td>{row["average_steps"]:.0f}</td><td>{row["average_reward"]:.1f}</td>'
                 f'<td>{row["average_invalid_actions"]:.0f}</td></tr>'
                 for i, (row, ok) in enumerate(zip(records, success)))
  title = f'Performance after {data["episodes"]:,} training episodes'
  document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Warehouse Robot · Evaluation</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1220;color:#e2e8f0;font:18px system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:52px 38px}h1{font-size:42px;line-height:1.15;margin:14px 0}
h2{font-size:25px;margin:0 0 12px}.eyebrow{color:#38bdf8;text-transform:uppercase;letter-spacing:3px;font-size:13px}
.muted{color:#94a3b8;line-height:1.6}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:32px 0}
.card,section{background:#131f32;border:1px solid #26374e;border-radius:16px;padding:24px}
.value{font-size:40px;font-weight:750;margin:10px 0}.label{font-size:14px;color:#94a3b8}
section{margin:20px 0}svg{width:100%;height:auto}svg text{fill:#94a3b8;font:12px system-ui}
.good{color:#4ade80}.bad{color:#fb7185}.split{display:flex;height:24px;border-radius:8px;overflow:hidden;margin:24px 0 12px}
button{background:#1e3a5f;border:1px solid #38bdf8;color:white;border-radius:8px;padding:10px 18px;cursor:pointer}
table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;padding:12px;border-bottom:1px solid #26374e}
details{overflow:auto}summary{cursor:pointer;padding:8px 0 20px}.controls{float:right}
@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}h1{font-size:30px}main{padding:24px}}
@media print{body{background:white;color:#111}.card,section{background:white;break-inside:avoid}.controls{display:none}}
</style><main><div class="controls"><button onclick="document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen()">Full screen</button></div>
<div class="eyebrow">Warehouse robot / unseen-map evaluation</div>'''
  document += f'<h1>{title}</h1><p class="muted">One greedy-policy episode on each of {count} unseen maps · Training seed {data["seed"]}</p>'
  cards = [(f'{summary["success_rate"]:.0%}', "Unseen-map success", f'{solved} solved / {count} tested'),
           (solved_average, "Average actions · solved maps", "Movement + pickup + drop-off"),
           (f'{summary["average_reward"]:.1f}', "Average reward · all maps", "Unshaped environment reward"),
           (f'{summary["average_invalid_actions"]:.1f}', "Invalid actions · per map", "Average over all tested maps")]
  document += '<div class="cards">' + ''.join(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="label">{detail}</div></div>' for value, label, detail in cards) + '</div>'
  document += f'<section><h2>Delivery outcome</h2><div class="split"><div style="width:{solved/count*100}%;background:#4ade80"></div><div style="flex:1;background:#fb7185"></div></div><span class="good">{solved} solved</span> &nbsp; / &nbsp; <span class="bad">{count-solved} failed</span><p class="muted">Average actions across all maps: {summary["average_steps"]:.1f}. Failures remain included in all-map averages.</p></section>'
  document += '<section><h2>Actions taken on each evaluation map</h2><p class="muted">Map index on the horizontal axis; action count on the vertical axis. Blue = solved; pink = failed. Hover over a bar for its value.</p>' + chart([r['average_steps'] for r in records], 'Evaluation map', successes=success) + '</section>'
  document += '<section>' + training_section + '</section>'
  document += '<section><h2>How to interpret this result</h2><p class="muted">The policy uses Q-learning with BFS nearest-target guidance. These results measure completion and efficiency on this test set; they do not establish globally shortest routes or guaranteed success on every map. Evaluation maps use seeds outside the training seed range. No exploration or Q-value updates are used during evaluation.</p></section>'
  document += '<section><details><summary>Per-map results and reproducible seeds</summary><table><thead><tr><th>Map</th><th>Seed</th><th>Outcome</th><th>Actions</th><th>Reward</th><th>Invalid</th></tr></thead><tbody>' + rows + '</tbody></table></details></section></main></html>'
  path = Path(destination)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(document, encoding="utf-8")
  return path
