from pathlib import Path

import pandas as pd


def load_csv(path):
    """Read the raw customer CSV. Do not change the file."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)
