# TODO: total refactor -> capitals etc...

# This file contains constants.. Mainly strings.
# It's never a good idea to have a string scattered in your code-base across different files, so just put them here
from pathlib import Path
from typing import List

RED: str = 'r'
GRN: str = 'g'
LABEL: str = 'label'

TFL_ID: int = 19  # The pixel value in the labelIds png images

# Column name
SEQ_IMAG: str = 'seq_imag'  # Serial number of the image
NAME: str = 'name'
IMAG_PATH: str = 'imag_path'
GTIM_PATH: str = 'gtim_path'
JSON_PATH: str = 'json_path'
X: str = 'x'
Y: str = 'y'
COLOR: str = 'color'

# Data CSV columns:
CSV_INPUT: List[str] = [SEQ_IMAG, NAME, IMAG_PATH, JSON_PATH, GTIM_PATH]
CSV_OUTPUT: List[str] = [SEQ_IMAG, NAME, IMAG_PATH, JSON_PATH, GTIM_PATH, X, Y, COLOR]

SEQ: str = 'seq'  # The image seq number -> for tracing back the original image
IS_TRUE: str = 'is_true'  # Is it a traffic light or not.
IGNOR: str = 'is_ignore'  # If it's an unusual crop (like two tfl's or only half etc.) that you can just ignor it and
# investigate the reason after
CROP_PATH: str = 'path'
X0: str = 'x0'  # The bigger x value (the right corner)
X1: str = 'x1'  # The smaller x value (the left corner)
Y0: str = 'y0'  # The smaller y value (the lower corner)
Y1: str = 'y1'  # The bigger y value (the higher corner)
COL: str = 'col'  # Color

RELEVANT_IMAGE_PATH: str = 'path'
ZOOM: str = 'zoom'  # If you zoomed in the picture, then by how much? (0.5. 0.25 etc.).

# CNN input CSV columns:
CROP_RESULT: List[str] = [SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COL]
ATTENTION_RESULT: List[str] = [RELEVANT_IMAGE_PATH, X, Y, ZOOM, COL]

# Files path
BASE_SNC_DIR: Path = Path.cwd().parent
DATA_DIR: Path = (BASE_SNC_DIR / 'mobileye-team-5-mobileye/data')
CROP_DIR: Path = DATA_DIR / 'crops'
ATTENTION_PATH: Path = DATA_DIR / 'attention_results'

ATTENTION_CSV_NAME: str = 'attention_results.csv'
CROP_CSV_NAME: str = 'crop_results.csv'


# File names (directories to be appended automatically)
TFLS_CSV: str = 'tfls.csv'
CSV_OUTPUT_NAME: str = 'results.csv'
