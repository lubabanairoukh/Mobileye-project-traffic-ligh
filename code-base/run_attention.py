import json
import argparse
import math
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict

from matplotlib.axes import Axes

# Internal imports... Should not fail
from consts import IMAG_PATH, JSON_PATH, NAME, SEQ_IMAG, X, Y, COLOR, RED, GRN, DATA_DIR, TFLS_CSV, CSV_OUTPUT, \
    SEQ, CROP_DIR, CROP_CSV_NAME, ATTENTION_RESULT, ATTENTION_CSV_NAME, ZOOM, RELEVANT_IMAGE_PATH, COL, ATTENTION_PATH, \
    CSV_INPUT, RADIUS
from misc_goodies import show_image_and_gt
from data_utils import get_images_metadata
from crops_creator import create_crops

import tqdm  # for the progress bar
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter
from PIL import Image
import matplotlib.pyplot as plt

# Additional imports
import cv2
import imutils
def find_tfl_lights(c_image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Detect candidates for TFL lights and return coordinates for red and green lights.
    :param c_image_path: Path to the image file.
    :param kwargs: Additional arguments.
    :return: Dictionary with keys 'x', 'y', 'col' containing lists for red and green light coordinates and colors.
    """

    # Load the image
    image = cv2.imread(c_image_path)

    # Prepare lists for red and green light coordinates
    x_red, y_red = [], []
    x_green, y_green = [],[]
    colors = []  # This list will store color values (RED, GRN)

    # Convert image to HSV color space for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for red (manual adjustment to capture the full red range)
    lower_red_manual1 = np.array([0, 50, 50])
    upper_red_manual1 = np.array([10, 255, 255])
    lower_red_manual2 = np.array([170, 50, 50])
    upper_red_manual2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colors
    mask_red1 = cv2.inRange(hsv, lower_red_manual1, upper_red_manual1)
    mask_red2 = cv2.inRange(hsv, lower_red_manual2, upper_red_manual2)
    red_thresh = cv2.bitwise_or(mask_red1, mask_red2)  # Combine both red ranges

    # Define the HSV range for green
    lower_green = np.array([32, 74, 64])
    upper_green = np.array([119, 255, 220])
    green_thresh = cv2.inRange(hsv, lower_green, upper_green)

    # Now, instead of finding circles or contours, we just look for non-zero pixels in the masks
    red_coords = np.column_stack(np.where(red_thresh > 0))  # Get all pixel coordinates where red is detected
    green_coords = np.column_stack(np.where(green_thresh > 0))  # Get all pixel coordinates where green is detected

    # Append red pixel coordinates
    for coord in red_coords:
        x_red.append(coord[1])  # X is the column index in the image
        y_red.append(coord[0])  # Y is the row index in the image
        colors.append(RED)

    # Append green pixel coordinates
    for coord in green_coords:
        x_green.append(coord[1])  # X is the column index in the image
        y_green.append(coord[0])  # Y is the row index in the image
        colors.append(GRN)

    # Combine red and green detections into one set of coordinates
    x_coords = x_red + x_green
    y_coords = y_red + y_green

    return {
        X: x_coords,
        Y: y_coords,
        COLOR: colors,
    }

def test_find_tfl_lights(row: Series, args: Namespace) -> DataFrame:
    """
    Run the attention code-base
    """
    image_path: str = row[IMAG_PATH]
    json_path: str = row[JSON_PATH]

    # Load the image
    image: np.ndarray = np.array(Image.open(image_path))

    if args.debug and json_path is not None:
        # Load ground truth data if available
        gt_data: Dict[str, Any] = json.loads(Path(json_path).read_text())
        what: List[str] = ['traffic light']
        objects: List[Dict[str, Any]] = [o for o in gt_data['objects'] if o['label'] in what]
        ax: Optional[Axes] = show_image_and_gt(image, objects, f"{row[SEQ_IMAG]}: {row[NAME]} GT")
    else:
        ax = None

    # Call your find_tfl_lights function
    attention_dict: Dict[str, Any] = find_tfl_lights(image_path, some_threshold=42, debug=args.debug)
    attention: DataFrame = pd.DataFrame(attention_dict)

    # Copy all image metadata from the row into the results
    for k, v in row.items():
        attention[k] = v

    if attention.empty:
        print(f"No detections in image: {image_path}")
        tfl_x = np.array([])
        tfl_y = np.array([])
        color = np.array([], dtype=str)
    else:
        tfl_x: np.ndarray = attention[X].values
        tfl_y: np.ndarray = attention[Y].values
        color: np.ndarray = attention[COLOR].values.astype(str)

    is_red = color == RED
    is_green = color == GRN

    print(f"Image: {image_path}, {is_red.sum()} reds, {is_green.sum()} greens..")

    if args.debug:
        # Plotting detections
        plt.figure(f"{row[SEQ_IMAG]}: {row[NAME]} detections")
        plt.clf()
        plt.subplot(211, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Original image.. Always try to compare your output to it')
        plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=4)
        plt.plot(tfl_x[is_green], tfl_y[is_green], 'g+', markersize=4)
        # Display the thresholded image (optional)
        plt.subplot(212, sharex=ax, sharey=ax)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title('Image with Detections')
        plt.suptitle("When you zoom on one, the other zooms too :-)")
    return attention

def prepare_list(in_csv_file: Path, args: Namespace) -> DataFrame:
    """
    We assume all students are working on the same CSV with files.
    This filters the list, so if you want to test some specific images, it's easy.
    This way, you can ask your friends how they performed on image 42 for example
    You may want to set the default_csv_file to anything on your computer, to spare the -f parameter.
    Note you will need different CSV files for attention and NN parts.
    The CSV must have at least columns: SEQ, NAME, TRAIN_TEST_VAL.
    """
    if args.image is not None:
        # Don't limit by count, take explicit images
        args.count = None

    csv_list: DataFrame = get_images_metadata(in_csv_file,
                                              max_count=args.count,
                                              take_specific=args.image)
    return pd.concat([pd.DataFrame(columns=CSV_INPUT), csv_list], ignore_index=True)


def run_on_list(meta_table: pd.DataFrame, func: callable, args: Namespace) -> pd.DataFrame:
    """
    Take a function, and run it on a list. Return accumulated results.

    :param meta_table: A DF with the columns your function requires
    :param func: A function to take a row of the DF, and return a DF with some results
    :param args:
    """
    acc: List[DataFrame] = []
    time_0: datetime = datetime.now()
    for _, row in tqdm.tqdm(meta_table.iterrows()):
        res: DataFrame = func(row, args)
        acc.append(res)
    time_1: datetime = datetime.now()
    all_results: DataFrame = pd.concat(acc).reset_index(drop=True)
    print(f"Took me {(time_1 - time_0).total_seconds()} seconds for "
          f"{len(all_results)} results from {len(meta_table)} files")

    return all_results


def save_df_for_part_2(crops_df: DataFrame, results_df: DataFrame):
    if not ATTENTION_PATH.exists():
        ATTENTION_PATH.mkdir()

    # Order the df by sequence, a nice to have.
    crops_sorted: DataFrame = crops_df.sort_values(by=SEQ)
    results_sorted: DataFrame = results_df.sort_values(by=SEQ_IMAG)

    attention_df: DataFrame = DataFrame(columns=ATTENTION_RESULT)
    row_template: Dict[str, Any] = {RELEVANT_IMAGE_PATH: '', X: '', Y: '', ZOOM: 0, COL: ''}
    for index, row in results_sorted.iterrows():
        row_template[RELEVANT_IMAGE_PATH] = row[IMAG_PATH]
        row_template[X], row_template[Y] = row[X], row[Y]
        row_template[COL] = row[COLOR]
        attention_df = attention_df._append(row_template, ignore_index=True)
    attention_df.to_csv(ATTENTION_PATH / ATTENTION_CSV_NAME, index=False)
    crops_sorted.to_csv(ATTENTION_PATH / CROP_CSV_NAME, index=False)


def parse_arguments(argv: Optional[Sequence[str]]):
    """
    Here are all the arguments in the attention stage.
    """
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=int, nargs='+', help='Specific image number(s) to run')
    parser.add_argument("-c", "--count", type=int, default=300, help="Max images to run")
    parser.add_argument('-f', '--in_csv_file', type=str, help='CSV file to read')
    parser.add_argument('-nd', '--no_debug', action='store_true', help='Show debug info')
    parser.add_argument('--attention_csv_file', type=str, help='CSV to write results to')

    args = parser.parse_args(argv)

    args.debug = not args.no_debug

    return args


def main(argv=None):
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """
    args: Namespace = parse_arguments(argv)
    default_csv_file: Path = DATA_DIR / TFLS_CSV
    csv_filename: Path = Path(args.in_csv_file) if args.in_csv_file else default_csv_file

    # This is your output CSV, look for CSV_OUTPUT const to see its columns.
    # No need to touch this function, if your curious about the result, put a break point and look at the result
    meta_table: DataFrame = prepare_list(csv_filename, args)
    print(f"About to run attention on {len(meta_table)} images. Watch out there!")

    # When you run your find find_tfl_lights, you want to add each time the output (x,y coordinates, color etc.)
    # to the output Dataframe, look at CSV_OUTPUT to see the names of the column and in what order.
    all_results: DataFrame = run_on_list(meta_table, test_find_tfl_lights, args)
    combined_df: DataFrame = pd.concat([pd.DataFrame(columns=CSV_OUTPUT), all_results], ignore_index=True)

    # make crops out of the coordinates from the DataFrame
   ## crops_df: DataFrame = create_crops(combined_df)

    # save the DataFrames in the right format for stage two.
   # save_df_for_part_2(crops_df, combined_df)
    #print(f"Got a total of {len(combined_df)} results")

    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()