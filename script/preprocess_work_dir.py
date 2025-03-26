import argparse
import json
import os
from pathlib import Path

import pandas as pd


def process_entry(entry, source_root, target_root, dataset) -> None:
    entry_id = f"{entry['id']}"
    csv_fnames = [Path(source_root) / f for f in entry['csv_path']]
    target_dir = Path(target_root) / entry_id

    target_dir.mkdir(parents=True, exist_ok=True)

    for csv_fname in csv_fnames:
        target_fname = target_dir / csv_fname.name
        if not target_fname.exists():  # Avoid overwriting existing symlinks
            os.symlink(csv_fname, target_fname)

    meta = {
        'source': dataset,
        'database_id': entry_id,
        'question': entry['question'],
        'dataset_names': [csv_fname.name for csv_fname in csv_fnames]
    }

    with open(target_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)


def process_dataset(dataset_name, source_root, target_root):
    source_path = Path(source_root) / dataset_name
    target_path = Path(target_root) / dataset_name

    df = pd.read_json(os.path.join(source_path, 'data.jsonl'), lines=True)
    df.apply(lambda x: process_entry(
        x, source_path, target_path, dataset_name), axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Process entries and create symbolic links and metadata.")

    parser.add_argument('--dataset', type=str, default='ALL',
                        help=(
                            'Dataset name, supported datasets: '
                            '[DACO, InsightBench, ALL] (case insensitive). '
                            'ALL processes both datasets.'
                        ))
    parser.add_argument('--target_root', type=str, default="",
                        help=(
                            'Path to the target root directory where '
                            'processed data will be saved'
                        ))
    args = parser.parse_args()

    datasets = ['daco', 'insightbench'] if args.dataset.lower() == 'all' else [
        args.dataset.lower()]

    source_root = Path(__file__).parent.parent / "data"
    if args.target_root == '':
        args.target_root = Path(__file__).parent.parent / "work"

    for dataset_name in datasets:
        process_dataset(dataset_name, source_root, args.target_root)


if __name__ == "__main__":
    main()
