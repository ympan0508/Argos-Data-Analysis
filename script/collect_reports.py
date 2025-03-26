import argparse
import glob
import json
from pathlib import Path


def collect_reports(path_pattern: str, output_file: str):
    reports = []

    files = glob.glob(path_pattern)

    for file_path in files:
        id_value = Path(file_path).parent.name

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
        except Exception as e:
            print(f"Unable to read file {file_path}: {e}")
            continue

        reports.append({
            'id': id_value,
            'report': report_content
        })

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Collect report files matching the path pattern and save them to "
            "a JSON file"
        ))
    parser.add_argument('--output_file', type=str, default=None,
                        help=(
                            'Path to the output JSON file, default: '
                            'work/output/daco_collected_reports.json'
                        ))
    parser.add_argument('--path_pattern', type=str, default=None,
                        help=(
                            'Path pattern to match, default: '
                            'work/daco/*/report.md'
                        ))

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = str(
            Path(__file__).parent.parent.resolve() /
            'work' / 'output' / 'daco_collected_reports.json'
        )

    if args.path_pattern is None:
        args.path_pattern = str(
            Path(__file__).parent.parent.resolve()
            / 'work' / 'daco' / '*' / 'report.md'
        )

    collect_reports(args.path_pattern, args.output_file)
