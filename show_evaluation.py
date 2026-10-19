"""Build and open a standalone evaluation dashboard without retraining."""
import argparse
import json
import webbrowser
from pathlib import Path

from warehouse_robot.evaluation_report import write_evaluation_report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--results', default='models/navigation_q.evaluation.json')
  parser.add_argument('--output', default=None)
  parser.add_argument('--no-open', action='store_true')
  args = parser.parse_args()
  source = Path(args.results)
  try:
    data = json.loads(source.read_text())
    path = write_evaluation_report(data, args.output or source.with_suffix('.html'))
  except (OSError, ValueError, KeyError) as error:
    parser.error(f'{error}. Run python train_q_learning.py to generate results first.')
  print(f'Evaluation dashboard: {path.resolve()}')
  if not args.no_open:
    webbrowser.open(path.resolve().as_uri())


if __name__ == '__main__':
  main()
