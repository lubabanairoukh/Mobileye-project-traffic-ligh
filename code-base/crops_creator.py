from typing import Dict, Any, Tuple, List

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, ZOOM, IMAG_PATH, CROP_WIDTH, CROP_HEIGHT, NAME, JSON_PATH, CSV_OUTPUT

from pandas import DataFrame

import numpy as np
from PIL import Image
from scipy.ndimage import zoom as nd_zoom
from scipy import ndimage


def make_crop(image: np.ndarray, x: float, y: float, crop_width: int = 40, crop_height: int = 60) -> Tuple[int, int, int, int, np.ndarray]:
    """
    Creates a crop from the image based on center coordinates and crop dimensions, 
    and handles cases where the crop goes out of image bounds.
    """
    x = int(round(x))
    y = int(round(y))

    half_width = crop_width // 2
    half_height = crop_height // 2

    # Ensure the crop stays within the image bounds, adjusting if necessary
    x0 = max(x - half_width, 0)
    x1 = min(x + half_width, image.shape[1])
    y0 = max(y - half_height, 0)
    y1 = min(y + half_height, image.shape[0])

    # Check if the crop goes out of bounds and add padding if necessary
    crop = image[y0:y1, x0:x1]

    # If the crop is smaller than expected (due to out-of-bounds), pad the image with black pixels
    if crop.shape[0] < crop_height or crop.shape[1] < crop_width:
        padded_crop = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
        padded_crop[:crop.shape[0], :crop.shape[1]] = crop  # Place the actual crop inside the padded image
        crop = padded_crop

    return x0, x1, y0, y1, crop


def check_crop(ground_truth: List[Dict[str, Any]], x0: int, x1: int, y0: int, y1: int) -> Tuple[bool, bool]:
    """
    Here you check if your crop contains a traffic light or not by checking with ground proof
    """
    crop_box = [x0, y0, x1, y1]
    for obj in ground_truth:
        if obj['label'] == 'traffic light':
            polygon = np.array(obj['polygon'])
            # Get bounding box of the polygon
            x_min = np.min(polygon[:, 0])
            x_max = np.max(polygon[:, 0])
            y_min = np.min(polygon[:, 1])
            y_max = np.max(polygon[:, 1])
            gt_box = [x_min, y_min, x_max, y_max]
            # Compute Intersection over Union (IoU)
            iou = compute_iou(crop_box, gt_box)
            if iou > 0.15:
                return True, False  # is_true=True, is_ignore=False
    return False, False  # is_true=False, is_ignore=False
    
def compute_iou(boxA, boxB):
    # Compute the intersection over union of two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def create_crops(df: DataFrame, ground_truths: Dict[str, List[Dict[str, Any]]], zoom_factor: float = 1.0) -> DataFrame:
    # Create the directory for crops if it doesn't exist
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # Initialize the result DataFrame with CROP_RESULT columns
    result_df = DataFrame(columns=CROP_RESULT)

    for index, row in df.iterrows():
        # Template for each row of results
        result_template: Dict[Any] = {
            SEQ: row[SEQ_IMAG], 
            IS_TRUE: '', 
            IGNOR: '', 
            CROP_PATH: '',
            X0: '', 
            X1: '', 
            Y0: '', 
            Y1: '', 
            COL: row[COLOR], 
            ZOOM: row['zoom_factor']
        }

        # Coordinates of the traffic light
        x = row[X]
        y = row[Y]

        # Load the image
        image = np.array(Image.open(row[IMAG_PATH]))

        # Define fixed crop size
        crop_width = CROP_WIDTH
        crop_height = CROP_HEIGHT

        # Get ground truth annotations for this image
        gt_annotations = ground_truths.get(row[IMAG_PATH], [])

        # Create the crop based on adjusted coordinates
        x0, x1, y0, y1, crop = make_crop(image, x, y, crop_width=crop_width, crop_height=crop_height)

        # Check if the crop contains a traffic light
        is_true, is_ignore = check_crop(gt_annotations, x0, x1, y0, y1)
        
        # Save the crop if it's valid
        if is_true:
            # Get zoom factor for this detection
            zoom_factor = row['zoom_factor']

            # Apply zoom to the image if needed
            if zoom_factor != 1.0:
                print(zoom_factor, row[SEQ_IMAG])
                # Zoom the image
                image_zoomed = ndimage.zoom(image, (zoom_factor, zoom_factor, 1), order=1)

                # Adjust coordinates due to zoom
                x *= zoom_factor
                y *= zoom_factor

                # Ensure that after zoom, the coordinates are still within image bounds
                x = min(max(x, 0), image_zoomed.shape[1])
                y = min(max(y, 0), image_zoomed.shape[0])

                # Create the crop again based on adjusted coordinates with zoom
                x0, x1, y0, y1, crop = make_crop(image_zoomed, x, y, crop_width=crop_width, crop_height=crop_height)

            # Save the crop image
            crop_filename = f"crop_{row[SEQ_IMAG]}_{index}.png"
            crop_path = CROP_DIR / crop_filename
            Image.fromarray(crop).save(crop_path)

            # Update the result template
            result_template[CROP_PATH] = crop_path.as_posix()
            result_template[X0], result_template[X1] = x0, x1
            result_template[Y0], result_template[Y1] = y0, y1
            result_template[IS_TRUE] = is_true
            result_template[IGNOR] = is_ignore

            # Append the result to the DataFrame
            result_df = result_df._append(result_template, ignore_index=True)

    return result_df