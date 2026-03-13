"""Download the Lichess puzzle database."""

import os
import csv
import requests
import zstandard as zstd

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "lichess_puzzles.csv")


def download_puzzles(output_path: str = RAW_CSV_PATH) -> str:
    """Download and decompress the Lichess puzzle CSV.

    The file is ~300MB compressed, ~1.5GB decompressed.
    Returns the path to the decompressed CSV.
    """
    if os.path.exists(output_path):
        print(f"Puzzle CSV already exists at {output_path}, skipping download.")
        return output_path

    print(f"Downloading puzzles from {PUZZLE_URL} ...")
    response = requests.get(PUZZLE_URL, stream=True)
    response.raise_for_status()

    decompressor = zstd.ZstdDecompressor()
    total_bytes = 0

    with open(output_path, "wb") as f_out:
        reader = decompressor.stream_reader(response.raw)
        while True:
            chunk = reader.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            f_out.write(chunk)
            total_bytes += len(chunk)
            if total_bytes % (100 * 1024 * 1024) < 1024 * 1024:
                print(f"  Decompressed {total_bytes / 1024 / 1024:.0f} MB ...")

    print(f"Done. Saved {total_bytes / 1024 / 1024:.1f} MB to {output_path}")
    return output_path


def iter_raw_puzzles(csv_path: str = RAW_CSV_PATH):
    """Iterate over the raw puzzle CSV, yielding one dict per row.

    Streams the file — does NOT load everything into memory.
    """
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        yield from reader


def load_raw_puzzles(csv_path: str = RAW_CSV_PATH) -> list[dict]:
    """Load the raw puzzle CSV into a list of dicts.

    Warning: loads all ~5.8M rows into memory (~6GB). Prefer
    iter_raw_puzzles() with reservoir sampling for large files.
    """
    return list(iter_raw_puzzles(csv_path))


if __name__ == "__main__":
    download_puzzles()
    puzzles = load_raw_puzzles()
    print(f"Loaded {len(puzzles)} puzzles.")
    if puzzles:
        print(f"Sample: {puzzles[0]}")
