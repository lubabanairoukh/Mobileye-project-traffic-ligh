from typing import Dict, Any, Tuple, List

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, ZOOM, IMAG_PATH

from pandas import DataFrame

import numpy as np
from PIL import Image


def make_crop(image: np.ndarray, x: float, y: float, crop_width: int = 40, crop_height: int = 60) -> Tuple[int, int, int, int, np.ndarray]:
    """
    The function that creates the crops from the image, now allowing for rectangular crops.
    'x0', 'x1' are the horizontal bounds, 'y0', 'y1' are the vertical bounds.
    """
    x = int(x)
    y = int(y)
    
    half_width = crop_width // 2
    half_height = crop_height // 2
    
    x0 = max(x - half_width, 0)
    x1 = min(x + half_width, image.shape[1])
    y0 = max(y - half_height, 0)
    y1 = min(y + half_height, image.shape[0])
    
    crop = image[y0:y1, x0:x1]  # Cropping the image
    return x0, x1, y0, y1, crop

def check_crop(ground_truth: List[Dict[str, Any]], x0: int, x1: int, y0: int, y1: int) -> Tuple[bool, bool]:
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that.
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
            if iou > 0.5:
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

def create_crops(df: DataFrame, ground_truths: Dict[str, Any]) -> DataFrame:
    # Create the directory for crops if it doesn't exist
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # Initialize the result DataFrame
    result_df = DataFrame(columns=CROP_RESULT)

    # Template for each row in the result DataFrame
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: '', ZOOM: 1}
    
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # Get the coordinates and image path
        x = row[X]
        y = row[Y]
        image_path = row[IMAG_PATH]

        # Load the image
        image = np.array(Image.open(image_path))  # Or use cv2.imread(image_path)

        # Define crop size (e.g., width = 40, height = 60 for rectangular crops)
        crop_width = 40
        crop_height = 60

        # Create the crop around the coordinates (with rectangular crop sizes)
        x0, x1, y0, y1, crop = make_crop(image, x, y, crop_width=crop_width, crop_height=crop_height)

        # Create a unique filename and save the crop
        crop_filename = f"crop_{row[SEQ_IMAG]}_{index}.png"
        crop_path = CROP_DIR / crop_filename
        Image.fromarray(crop).save(crop_path)

        # Get the ground truth for this image
        gt_annotations = ground_truths.get(row[IMAG_PATH], [])

        # Check if the crop contains a traffic light
        is_true, is_ignore = check_crop(gt_annotations, x0, x1, y0, y1)

        # Fill the result template with crop info
        result_template[X0], result_template[X1] = x0, x1
        result_template[Y0], result_template[Y1] = y0, y1
        result_template[CROP_PATH] = crop_path.as_posix()
        result_template[IS_TRUE] = is_true
        result_template[IGNOR] = is_ignore

        # Append the result to the DataFrame
        result_df = result_df._append(result_template, ignore_index=True)
    
    return result_df
