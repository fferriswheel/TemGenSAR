#!/usr/bin/env python3
import argparse
from pathlib import Path

from prepare_gensar_lite import load_item_ids, parse_sid_specs, write_sid_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sid-spec", action="append", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    item_ids = load_item_ids(data_dir)
    asin_to_int = {asin: index + 1 for index, asin in enumerate(item_ids)}
    written = write_sid_files(
        Path(args.output_dir), parse_sid_specs(args.sid_spec), asin_to_int
    )
    for name, files in written.items():
        print(
            f"{name}: {files['semantic_index_file']}, "
            f"{files['collaborate_index_file']}"
        )


if __name__ == "__main__":
    main()
