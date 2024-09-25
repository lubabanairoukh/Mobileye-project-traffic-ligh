import json
import argparse
from datetime import datetime
from argparse import Namespace
from pathlib import Path
from typing import Sequence, Optional, List, Any, Dict, Tuple

from matplotlib.axes import Axes

# Internal imports... Should not fail
from consts import IMAG_PATH, JSON_PATH, NAME, SEQ_IMAG, X, Y, COLOR, RED, GRN, DATA_DIR, TFLS_CSV, CSV_OUTPUT, \
    SEQ, CROP_DIR, CROP_CSV_NAME, ATTENTION_RESULT, ATTENTION_CSV_NAME, ZOOM, RELEVANT_IMAGE_PATH, COL, ATTENTION_PATH, \
    CSV_INPUT, RADIUS, BASE_RADIUS
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
from scipy.spatial import distance

from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from typing import List, Dict, Tuple


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize the brightness.
    :param image: Input image.
    :return: Image after CLAHE has been applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_gray = clahe.apply(gray)
    return cv2.cvtColor(normalized_gray, cv2.COLOR_GRAY2BGR)


def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to the image to reduce noise.
    :param image: Input image.
    :return: Blurred image.
    """
    res = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow("Green Mask", res)
    # Wait for key press to proceed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 


def apply_sobel_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection to the grayscale image.
    :param gray_image: Grayscale image.
    :return: Edge-detected image.
    """
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
    grad = cv2.magnitude(grad_x, grad_y)  # Combine the gradients

    return cv2.convertScaleAbs(grad)


def create_color_masks(hsv_image: np.ndarray) -> np.ndarray:
    """
    Creates binary masks for detecting specific colors (white, blue, green, and red) in the HSV image.
    :param hsv_image: HSV image.
    :return: Combined binary mask for white, blue, green, and red regions.
    """
    # White center detection
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    # Blue surround detection
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Green light detection
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Red light detection (two ranges due to hue wrapping)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Combine the white, blue, green, and red masks
    mask_combined = cv2.bitwise_or(cv2.bitwise_or(mask_white, mask_blue), mask_green)
    mask_combined = cv2.bitwise_or(mask_combined, mask_red)

    return mask_combined


# Check if the region of interest is dark
def is_dark_area(roi: np.ndarray, threshold: int = 50) -> bool:
    """
    Check if the region of interest (ROI) is dark based on pixel intensity.
    :param roi: The region of interest (sub-image).
    :param threshold: Pixel intensity threshold to consider the area dark.
    :return: True if the area is considered dark, False otherwise.
    """
    return np.mean(roi) < threshold


def is_circular(contour: np.ndarray, tolerance: float = 0.5) -> bool:
    """
    Checks if a given contour is approximately circular.
    :param contour: Contour points.
    :param tolerance: Allowable tolerance for deviations from a perfect circle (default: 0.2).
    :return: True if the contour is circular, False otherwise.
    """
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False  # Avoid division by zero
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return 1 - tolerance <= circularity <= 1 + tolerance


def find_light_contours(image: np.ndarray, mask: np.ndarray, gray_image: np.ndarray, min_brightness: int = 100) -> Tuple[List[int], List[int], List[str], List[float]]:
    """
    Finds contours of possible traffic lights using both color detection and circularity check.
    :param image: Input image.
    :param mask: Binary mask for detecting lights.
    :param gray_image: Grayscale image for brightness and darkness checks.
    :param min_brightness: Minimum brightness threshold for detecting lights.
    :return: Lists of x and y coordinates of detected lights, their associated colors, and radii.
    """
    # Find contours from the mask (color-based detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to store coordinates, colors, and radii of detected lights
    x_coords, y_coords, colors, radii = [], [], [], []

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 2000 and is_circular(contour, tolerance=0.2):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 1.2:  # Filter for near-circular shapes
                # Calculate radius
                radius = np.sqrt(area / np.pi)
                radii.append(radius)
                # Append center coordinates
                x_coords.append(x + w // 2)
                y_coords.append(y + h // 2)
                # Determine color (implement your logic here)
                colors.append('green' or 'red')  # Replace with actual color detection logic
    return x_coords, y_coords, colors, radii


def group_close_detections(x_coords: List[int], y_coords: List[int], colors: List[str], radii: List[float], threshold: int = 20) -> Tuple[List[int], List[int], List[str], List[float]]:
    """
    Groups detections that are too close together based on a distance threshold.
    The threshold can be made more strict to avoid creating multiple crops of the same light.
    """
    if len(x_coords) <= 1:
        return x_coords, y_coords, colors, radii  # Nothing to group if only 1 or no detections

    grouped_x, grouped_y, grouped_colors, grouped_radii = [], [], [], []
    points = np.array(list(zip(x_coords, y_coords)))
    grouped = [False] * len(points)

    for i, point in enumerate(points):
        if grouped[i]:
            continue
        current_group = [i]
        for j, other_point in enumerate(points):
            if i != j and not grouped[j]:
                dist = distance.euclidean(point, other_point)
                if dist < threshold:
                    current_group.append(j)

        # Calculate the average position and radius for the group
        avg_x = int(np.mean([x_coords[idx] for idx in current_group]))
        avg_y = int(np.mean([y_coords[idx] for idx in current_group]))
        group_color = colors[current_group[0]]  # Assuming same color
        avg_radius = np.mean([radii[idx] for idx in current_group])

        # Add the grouped detection to the final list
        grouped_x.append(avg_x)
        grouped_y.append(avg_y)
        grouped_colors.append(group_color)
        grouped_radii.append(avg_radius)

        # Mark all members of the group as grouped
        for idx in current_group:
            grouped[idx] = True

    return grouped_x, grouped_y, grouped_colors, grouped_radii


# Helper function for visualization (for debugging purposes)
def visualize_detections(image: np.ndarray, x_coords: List[int], y_coords: List[int], colors: List[str]):
    """
    Visualizes the detected traffic lights by drawing circles at the detected coordinates.
    :param image: Original image.
    :param x_coords: List of x coordinates of detected lights.
    :param y_coords: List of y coordinates of detected lights.
    :param colors: List of colors corresponding to detected lights ('red', 'green').
    """
    for (x, y, color) in zip(x_coords, y_coords, colors):
        if color == 'red':
            cv2.circle(image, (x, y), 20, (0, 0, 255), 2)  # Red circle
        elif color == 'green':
            cv2.circle(image, (x, y), 20, (0, 255, 0), 2)  # Green circle

    # Display the image with detections
    cv2.imshow('Detected Traffic Lights', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_tfl_lights(c_image_path: str, **kwargs) -> Dict[str, Any]:
    """
    Main function that detects traffic lights by combining color and edge detection techniques.
    :param c_image_path: Path to the image file.
    :param kwargs: Additional arguments (e.g., for debugging).
    :return: Dictionary with keys 'x', 'y', 'col' containing lists for detected light coordinates and colors.
    """
    # Load the image
    image = cv2.imread(c_image_path)

    # Apply CLAHE for brightness normalization
    normalized_image = apply_clahe(image)

    # Use color detection
    hsv = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
    mask_combined = create_color_masks(hsv)

    # Convert the normalized image to grayscale
    gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)

    # Find light contours using the color mask and grayscale image
    x_coords, y_coords, colors, radii = find_light_contours(image, mask_combined, gray)

    # In your find_tfl_lights function
    grouped_x, grouped_y, grouped_colors, grouped_radii = group_close_detections(x_coords, y_coords, colors, radii, threshold=30)

    # Return the grouped coordinates, colors, and radii
    return {
        'x': grouped_x,
        'y': grouped_y,
        'col': grouped_colors,
        'radius': grouped_radii,
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

    ## Call your find_tfl_lights function
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
        # Enhanced Debug Visualization
        plt.figure(f"{row[SEQ_IMAG]}: {row[NAME]} detections")
        plt.clf()

        # Plot 1: Original image with traffic light detections
        plt.subplot(211, sharex=ax, sharey=ax)
        plt.imshow(image)
        plt.title('Original image.. Compare with detections below')
        plt.plot(tfl_x[is_red], tfl_y[is_red], 'rx', markersize=6, label='Red lights')
        plt.plot(tfl_x[is_green], tfl_y[is_green], 'g+', markersize=6, label='Green lights')
        plt.plot(tfl_x[~is_red & ~is_green], tfl_y[~is_red & ~is_green], 'bo', markersize=6, label='Other lights')

        plt.legend(loc='upper left')

        # Plot 2: Image with colored circles on detections
        plt.subplot(212, sharex=ax, sharey=ax)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title('Image with detected traffic lights')
        
        plt.tight_layout()  # Better layout

        # Show the images without blocking the flow of execution
        plt.show(block=True)
        
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
    Take a function and run it on a list. Return accumulated results.
    Parallelize the function calls for performance.
    """
    acc: List[DataFrame] = []
    time_0: datetime = datetime.now()

    # Use ProcessPoolExecutor to parallelize the work
    with ProcessPoolExecutor() as executor:
        # Map each row of the meta_table to the function
        futures = [executor.submit(func, row, args) for _, row in meta_table.iterrows()]

        # Collect the results as they complete
        for future in tqdm.tqdm(futures):
            try:
                res: DataFrame = future.result()
                acc.append(res)
            except Exception as e:
                print(f"Error processing row: {e}")

    time_1: datetime = datetime.now()
    all_results: DataFrame = pd.concat(acc).reset_index(drop=True)
    print(f"Took me {(time_1 - time_0).total_seconds()} seconds for "
          f"{len(all_results)} results from {len(meta_table)} files")

    return all_results


def save_df_to_csv(crops_df: DataFrame, results_df: DataFrame):
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


def determine_zoom_factor(radius: float, base_radius: float = 10.0) -> float:
    """
    Determines zoom factor based on the radius of the detected circle.
    Reverses the zoom logic so that larger circles are zoomed in more, 
    and smaller circles are zoomed in less.
    :param radius: Detected radius of the traffic light.
    :param base_radius: The standard radius size you want in your crops.
    :return: Calculated zoom factor.
    """
    if radius <= 0 or np.isnan(radius):
        return 1.0  # Default zoom factor to 1.0 if the radius is invalid
    
    zoom_factor = BASE_RADIUS / radius
    # Limit the zoom factor to prevent extreme zooms (optional)
    #zoom_factor = min(max(0.5, zoom_factor), 2.0)
    return zoom_factor


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

    # Calculate zoom factors based on detected radii and the base radius
    combined_df['zoom_factor'] = combined_df['radius'].apply(determine_zoom_factor)

    # Load ground truth data for each image
    ground_truths = load_ground_truth(meta_table)

    # make crops out of the coordinates from the DataFrame
    crops_df: DataFrame = create_crops(combined_df, ground_truths)

    # save the DataFrames in the right format for stage two.
    save_df_to_csv(crops_df, combined_df)
    print(f"Got a total of {len(combined_df)} results")

    if args.debug:
        plt.show(block=True)


if __name__ == '__main__':
    main()