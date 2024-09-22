from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from pandas import DataFrame
import torch


from consts import SEQ_IMAG, NAME, IMAG_PATH, GTIM_PATH, JSON_PATH, CSV_OUTPUT
from manual_comments import blacklisted
from misc_goodies import temp_seed

pd.set_option('display.width', 200, 'display.max_rows', 200,
              'display.max_columns', 200, 'max_colwidth', 40)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def get_images_metadata(csv_path: Path,
                        max_count: Optional[int] = None,
                        take_specific: Optional[int] = None,
                        random_seed=0) -> DataFrame:
    """
    Get the images from the CSV file. Rebuild their path to absolute path.
    """
    df_1: DataFrame = pd.read_csv(csv_path)
    df: DataFrame = df_1[[k for k in df_1 if 'Unnamed' not in k]]

    if take_specific is not None:
        # Expecting a list of ints. Overrides anything else, including blacklisting
        if not np.iterable(take_specific):
            take_specific: List[int] = [take_specific]
        df: DataFrame = df[df[SEQ_IMAG].isin(take_specific)]
    else:
        # You can remove images with not a single relevant TFL here
        df: DataFrame = df[~df[NAME].isin(blacklisted)]

        if max_count is not None:

            if abs(max_count) < len(df):
                # Choose "randomly" only that amount of files:
                if max_count < 0:
                    # Take first images:
                    df: DataFrame = df.iloc[:-max_count]
                else:
                    # Take "Random" images
                    with temp_seed(random_seed):
                        x: np.ndarray = np.random.choice(len(df), max_count)
                        df: DataFrame = df.iloc[x]

    # Make all paths absolute:
    csv_dir: Path = csv_path.parent
    df: DataFrame = df.reset_index(drop=True)  # Avoid the usual warning
    for res_type in [IMAG_PATH, GTIM_PATH, JSON_PATH]:
        df[res_type] = [(csv_dir / x).resolve().as_posix() for x in df[res_type]]
    return df
