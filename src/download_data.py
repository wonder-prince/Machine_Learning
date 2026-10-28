"""Download the exact UCI bank-additional-full.csv used by the manual."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "bank-additional-full.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        print(f"Already present: {DESTINATION}")
        return
    with urlopen(URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    data = archive.read("bank-additional/bank-additional-full.csv")
    DESTINATION.write_bytes(data)
    print(f"Downloaded {DESTINATION} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
