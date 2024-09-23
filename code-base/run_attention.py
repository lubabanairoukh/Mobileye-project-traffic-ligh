# This file contains the skeleton you can use for traffic light attention
import json
import argparse
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict

from matplotlib.axes import Axes

# Internal imports... Should not fail
from consts import IMAG_PATH, JSON_PATH, NAME, SEQ_IMAG, X, Y, COLOR, RED, GRN, DATA_DIR, TFLS_CSV, CSV_OUTPUT, \
    SEQ, CROP_DIR, CROP_CSV_NAME, ATTENTION_RESULT, ATTENTION_CSV_NAME, ZOOM, RELEVANT_IMAGE_PATH, COL, ATTENTION_PATH, \
    CSV_INPUT
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
from matplotlib import colors


def find_tfl_lights(c_image: np.ndarray, **kwargs) -> Dict[str, Any]:
    """
    Detect candidates for traffic lights using color thresholding and connected component analysis.

    :param c_image: a H*W*3 RGB image of dtype np.float32 (values in 0-1 range).
    :param kwargs: Additional arguments.
    :return: Dictionary with keys 'x', 'y', 'col', each containing a list.
    """

    # Convert RGB image to HSV color space
    hsv_image = colors.rgb_to_hsv(c_image)

    # Extract the Hue, Saturation, and Value channels
    hue = hsv_image[:, :, 0]
    sat = hsv_image[:, :, 1]
    val = hsv_image[:, :, 2]

    # Define thresholds for red and green colors
    # Red color thresholds (Hue around 0 and 1)
    red_mask = (((hue < 0.05) | (hue > 0.95)) & (sat > 0.5) & (val > 0.5))

    # Green color thresholds (Hue around 0.33)
    green_mask = ((hue > 0.25) & (hue < 0.4) & (sat > 0.5) & (val > 0.5))

    # Apply morphological operations to remove small noises
    structure_element = ndimage.generate_binary_structure(2, 1)
    red_mask = ndimage.binary_opening(red_mask, structure_element)
    green_mask = ndimage.binary_opening(green_mask, structure_element)

    # Label connected components
    labeled_red, num_features_red = ndimage.label(red_mask)
    labeled_green, num_features_green = ndimage.label(green_mask)

    x_red = []
    y_red = []
    x_green = []
    y_green = []

    # For red components
    for label in range(1, num_features_red + 1):
        indices = np.argwhere(labeled_red == label)
        if indices.size == 0:
            continue
        y_coords, x_coords = indices[:, 0], indices[:, 1]
        x_mean = x_coords.mean()
        y_mean = y_coords.mean()
        x_red.append(x_mean)
        y_red.append(y_mean)

    # For green components
    for label in range(1, num_features_green + 1):
        indices = np.argwhere(labeled_green == label)
        if indices.size == 0:
            continue
        y_coords, x_coords = indices[:, 0], indices[:, 1]
        x_mean = x_coords.mean()
        y_mean = y_coords.mean()
        x_green.append(x_mean)
        y_green.append(y_mean)

    return {
        X: x_red + x_green,
        Y: y_red + y_green,
        COLOR: [RED] * len(x_red) + [GRN] * len(x_green),
    }


def test_find_tfl_lights(row: Series, args: Namespace) -> DataFrame:
    """
    Run the attention code-base
    """
    image_path: str = row[IMAG_PATH]
    json_path: str = row[JSON_PATH]
    image: np.ndarray = np.array(Image.open(image_path), dtype=np.float32) / 255

    if args.debug and json_path is not None:
        # This code-base demonstrates the fact you can read the bounding polygons from the json files
        # Then plot them on the image. Try it if you think you want to. Not a must...
        gt_data: Dict[str, Any] = json.loads(Path(json_path).read_text())
        what: List[str] = ['traffic light']
        objects: List[Dict[str, Any]] = [o for o in gt_data['objects'] if o['label'] in what]
        ax: Optional[Axes] = show_image_and_gt(image, objects, f"{row[SEQ_IMAG]}: {row[NAME]} GT")
    else:
        ax = None

    # In case you want, you can pass any parameter to find_tfl_lights, because it uses **kwargs
    attention_dict: Dict[str, Any] = find_tfl_lights(image, some_threshold=42, debug=args.debug)
    attention: DataFrame = pd.DataFrame(attention_dict)

    # Copy all image metadata from the row into the results, so we can track it later
    for k, v in row.items():
        attention[k] = v

    tfl_x: np.ndarray = attention[X].values
    tfl_y: np.ndarray = attention[Y].values
    color: np.ndarray = attention[COLOR].values
    is_red = color == RED
    is_green = color == GRN

    print(f"Image: {image_path}, {is_red.sum()} reds, {is_green.sum()} greens")

    if args.debug:
        # Show the image with the detections
        plt.figure(f"{row[SEQ_IMAG]}: {row[NAME]} detections")
        plt.clf()
        plt.subplot(211, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Original image.. Always try to compare your output to it')
        plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=4)
        plt.plot(tfl_x[~is_red], tfl_y[~is_red], 'g+', markersize=4)

        # Show clear image
        plt.subplot(212, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Some useless image for you')
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
    row_template: Dict[str, Any] = {RELEVANT_IMAGE_PATH: '', X: '', Y: '', ZOOM: 1, COL: ''}
    for index, row in results_sorted.iterrows():
        row_template[RELEVANT_IMAGE_PATH] = row[IMAG_PATH]
        # row_template[ZOOM] = results_sorted['zoom_value or whatever'] # TODO: this will break
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

def load_ground_truth(meta_table):
    """
    Load ground truth annotations for each image in the meta table.
    This could be reading from JSON files that contain traffic light annotations.
    """
    ground_truths = {}
    for _, row in meta_table.iterrows():
        json_path = row[JSON_PATH]
        if json_path is not None:
            gt_data = json.loads(Path(json_path).read_text())
            ground_truths[row[IMAG_PATH]] = gt_data['objects']
    return ground_truths

def calculate_bbox_size(polygon: np.ndarray) -> int:
    """
    Given a polygon (bounding box), calculate the width, height, and area of the box.
    :param polygon: List of coordinates of the bounding box
    :return: Area (width * height) of the bounding box in pixels
    """
    x_min = np.min(polygon[:, 0])
    x_max = np.max(polygon[:, 0])
    y_min = np.min(polygon[:, 1])
    y_max = np.max(polygon[:, 1])
    
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    
    return width, height, area

def find_min_max_traffic_light_size(ground_truths: Dict[str, List[Dict[str, any]]]) -> Dict[str, float]:
    """
    Find the minimum and maximum sizes of traffic lights in the dataset using ground truth data.
    :param ground_truths: Dictionary with image paths as keys and ground truth annotations as values
    :return: Dictionary with min and max size (area), and typical width and height
    """
    min_size = float('inf')
    max_size = float('-inf')
    sizes = []

    for image_path, objects in ground_truths.items():
        for obj in objects:
            if obj['label'] == 'traffic light':
                polygon = np.array(obj['polygon'])
                width, height, area = calculate_bbox_size(polygon)
                sizes.append(area)
                min_size = min(min_size, area)
                max_size = max(max_size, area)
    
    avg_width = np.mean([calculate_bbox_size(np.array(obj['polygon']))[0] for obj_list in ground_truths.values() for obj in obj_list if obj['label'] == 'traffic light'])
    avg_height = np.mean([calculate_bbox_size(np.array(obj['polygon']))[1] for obj_list in ground_truths.values() for obj in obj_list if obj['label'] == 'traffic light'])
    
    return {
        'min_size': min_size,
        'max_size': max_size,
        'avg_width': avg_width,
        'avg_height': avg_height
    }

def determine_zoom_factor(size: float, typical_size: float) -> float:
    """
    Calculate the zoom factor based on the size of the traffic light and the typical size.
    :param size: Area of the detected traffic light bounding box
    :param typical_size: Average or typical traffic light size in the dataset
    :return: Suggested zoom factor
    """
    return typical_size / size

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

    # Load ground truth data for each image
    ground_truths = load_ground_truth(meta_table)

    # Find the minimum, maximum, and average sizes of traffic lights in the dataset
    size_stats = find_min_max_traffic_light_size(ground_truths)

    # For each image, determine the zoom factor based on the traffic light size
    for image_path, objects in ground_truths.items():
        for obj in objects:
            if obj['label'] == 'traffic light':
                polygon = np.array(obj['polygon'])
                _, _, area = calculate_bbox_size(polygon)
                zoom_factor = determine_zoom_factor(area, size_stats['avg_width'] * size_stats['avg_height'])
                print(f"Zoom factor for {image_path}: {zoom_factor}")

    # make crops out of the coordinates from the DataFrame
    crops_df: DataFrame = create_crops(combined_df, ground_truths)

    # If you entered the zoom variable in the create_crops func, then extract it variable from the crops_df put it in
    # the combined_df.

    # save the DataFrames in the right format for stage two.
    save_df_for_part_2(crops_df, combined_df)
    print(f"Got a total of {len(combined_df)} results")

    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()
