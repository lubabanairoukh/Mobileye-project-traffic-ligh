import json
import argparse
import math
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict, Tuple

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
from matplotlib import colors

# Additional imports
import cv2
import imutils


def visualize_detections(image: np.ndarray, x_coords: List[int], y_coords: List[int], cutoff_y: int):
    """
    Visualizes the detected traffic lights by drawing circles at the detected coordinates.
    :param image: Original image.
    :param x_coords: List of x coordinates of detected lights.
    :param y_coords: List of y coordinates of detected lights.
    :param cutoff_y: y-coordinate of the cutoff line.
    """
    # Draw circles at detected points
    for (x, y) in zip(x_coords, y_coords):
        cv2.circle(image, (x, y), 20, (255, 0, 0), 2)  # Blue circle
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Red center point
    
    # Show the image with detections
    cv2.imshow('Detected Traffic Lights', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize the brightness.
    :param image: Input grayscale image.
    :return: Image after CLAHE has been applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_gray = clahe.apply(gray)
    return cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)


def create_color_masks(hsv_image: np.ndarray) -> np.ndarray:
    """
    Creates binary masks for detecting specific colors (white, blue, and green) in the HSV image.
    :param hsv_image: HSV image.
    :return: Combined binary mask for white, blue, and green regions.
    """
    # White center detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # Blue surround detection
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # Adjusted Green light detection
    lower_green = np.array([30, 50, 50])  # Adjusted to better detect green lights
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Combine the white, blue, and green masks
    mask_combined = cv2.bitwise_or(cv2.bitwise_or(mask_white, mask_blue), mask_green)
    
    return mask_combined


def apply_morphological_operations(mask: np.ndarray) -> np.ndarray:
    """
    Applies morphological operations (closing) to the mask to connect nearby regions.
    :param mask: Binary mask.
    :return: Morphologically transformed mask.
    """
    # Use morphological closing to connect regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask_morphed


def is_dark_area(roi: np.ndarray, threshold: int = 50) -> bool:
    """
    Check if the region of interest (ROI) is dark based on pixel intensity.
    :param roi: The region of interest (sub-image).
    :param threshold: Pixel intensity threshold to consider the area dark.
    :return: True if the area is considered dark, False otherwise.
    """
    return np.mean(roi) < threshold


def find_light_contours(mask: np.ndarray, gray_image: np.ndarray, min_brightness: int = 100) -> Tuple[List[int], List[int], List[str]]:
    """
    Finds contours of possible traffic lights, filters them based on size and shape,
    and checks if there's a dark area above or below the detected light.
    :param mask: Binary mask for detecting lights.
    :param gray_image: Grayscale image to analyze halos and dark areas.
    :param min_brightness: Minimum brightness threshold for detecting lights.
    :return: Lists of x and y coordinates of detected lights and their associated colors.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x_coords, y_coords, colors = [], [], []
    
    for contour in contours:
        # Filter based on area
        area = cv2.contourArea(contour)
        if 100 < area < 2000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.2:
                # Check brightness in grayscale
                roi = gray_image[max(0, y-10):min(gray_image.shape[0], y+h+10), max(0, x-10):min(gray_image.shape[1], x+w+10)]
                brightness = np.mean(roi)
                if brightness > min_brightness:
                    # Check for dark area below (for green) or above (for red)
                    below_roi = gray_image[y+h:min(gray_image.shape[0], y+2*h), x:x+w]  # Below detected light
                    above_roi = gray_image[max(0, y-h):y, x:x+w]  # Above detected light
                    
                    if is_dark_area(below_roi):  # Light with dark area below (likely green)
                        x_coords.append(x + w // 2)
                        y_coords.append(y + h // 2)
                        colors.append('green')
                    elif is_dark_area(above_roi):  # Light with dark area above (likely red)
                        x_coords.append(x + w // 2)
                        y_coords.append(y + h // 2)
                        colors.append('red')
    
    return x_coords, y_coords, colors


def load_and_preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the image, crops it, and converts it to grayscale and HSV color spaces.
    :param image_path: Path to the image file.
    :return: Cropped image, grayscale image, and HSV image.
    """
    # Load the image
    image = cv2.imread(image_path)
    
    # Get image dimensions and crop the bottom 60%
    height, _ = image.shape[:2]
    cutoff_y = int(height * 0.4)
    image = image[:cutoff_y, :]
    
    # Convert to grayscale and HSV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    return image, gray, hsv


def find_tfl_lights(c_image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Main function that detects traffic lights by combining image processing techniques
    such as color masking, morphological operations, and contour detection.
    :param c_image_path: Path to the image file.
    :param kwargs: Additional arguments (e.g., for debugging).
    :return: Dictionary with keys 'x', 'y', 'col' containing lists for detected light coordinates and colors.
    """
    # Load the image
    image = cv2.imread(c_image_path)
    
    # Apply CLAHE for brightness normalization
    normalized_image = apply_clahe(image)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
    
    # Create color masks for white, blue, and green regions (traffic light candidates)
    mask_combined = create_color_masks(hsv)
    
    # Convert the normalized image to grayscale
    gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    
    # Find contours and filter them to detect traffic lights
    x_coords, y_coords, colors = find_light_contours(mask_combined, gray, min_brightness=80)  # Reduced brightness threshold
    
    # Visualization for debugging (optional)
    if kwargs.get('debug', False):
        height, width = image.shape[:2]
        cutoff_y = int(height * 0.4)
        visualize_detections(image, x_coords, y_coords, cutoff_y)
    
    # Return the coordinates and the color (blue and green)
    return {
        'x': x_coords,
        'y': y_coords,
        'col': colors,
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
        tfl_x: np.ndarray = attention['x'].values
        tfl_y: np.ndarray = attention['y'].values
        color: np.ndarray = attention['col'].values.astype(str)  # Changed from 'color' to 'col'

    is_red = color == RED
    is_green = color == GRN

    print(f"Image: {image_path}, {is_red.sum()} reds, {is_green.sum()} greens, {(~is_red & ~is_green).sum()} others")

    if args.debug:
        # Plotting detections
        plt.figure(f"{row[SEQ_IMAG]}: {row[NAME]} detections")
        plt.clf()
        plt.subplot(211, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Original image.. Always try to compare your output to it')
        plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=4)
        plt.plot(tfl_x[is_green], tfl_y[is_green], 'g+', markersize=4)
        plt.plot(tfl_x[~is_red & ~is_green], tfl_y[~is_red & ~is_green], 'bo', markersize=4)  # Plot other colors
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
    #crops_df: DataFrame = create_crops(combined_df, ground_truths)

    # save the DataFrames in the right format for stage two.
    #save_df_for_part_2(crops_df, combined_df)
    print(f"Got a total of {len(combined_df)} results")

    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()