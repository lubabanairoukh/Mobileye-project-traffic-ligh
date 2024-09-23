## CityScape Traffic Light Detection and Verification

### Data

For details about cityscape, see https://www.cityscapes-dataset.com/dataset-overview/
To download after login: https://www.cityscapes-dataset.com/downloads/
Need to download:
- gtFine_trainvaltest.zip (241MB)
- leftImg8bit_trainvaltest.zip (11GB)  
(Can also download the leftImg8bit_trainextra.zip (44GB), but it's pretty big...)

### Local arrangement of the files
All paths are absolute relative to the CSV folder.  



# Traffic Light Detection - `find_tfl_lights` Function

This function, `find_tfl_lights`, detects potential traffic light signals (red and green) in an image. It uses color segmentation to identify the regions of interest and returns the coordinates of the detected lights along with their colors.

## Steps of the Function

### 1. **Reading the Image**
   ```python
   image = cv2.imread(c_image_path)
   ```
   **What it does**:  
   - The function reads the input image from the provided file path.
   - This image is used as the input for detecting red and green traffic lights.

   **Why**:  
   - The image is required to process and extract the regions where traffic lights might be present.

### 2. **Color Conversion to HSV**
   ```python
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   ```
   **What it does**:  
   - Converts the image from BGR (Blue, Green, Red) to HSV (Hue, Saturation, Value) color space.  
   
   **Why**:  
   - HSV color space allows easier color segmentation (especially for traffic light colors) because the hue value can separate color information more effectively than BGR.

### 3. **Defining Color Ranges for Red and Green Lights**
   ```python
   lower_red1 = np.array([0, 100, 100])
   upper_red1 = np.array([10, 255, 255])
   lower_red2 = np.array([160, 100, 100])
   upper_red2 = np.array([179, 255, 255])

   lower_green = np.array([40, 100, 100])
   upper_green = np.array([80, 255, 255])
   ```
   **What it does**:  
   - Defines the range of color values in HSV space that correspond to red and green lights.
   - Since red exists in two different hue ranges (due to HSV's circular nature), two ranges are used for red detection.

   **Why**:  
   - Proper color detection requires defining HSV ranges. These ranges help isolate only the pixels that represent red or green colors in the image.

### 4. **Thresholding to Isolate Red and Green Colors**
   ```python
   mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
   mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
   red_thresh = cv2.bitwise_or(mask_red1, mask_red2)

   green_thresh = cv2.inRange(hsv, lower_green, upper_green)
   ```
   **What it does**:  
   - Creates binary masks for the red and green regions of the image.
   - For red: Combines two masks (`mask_red1` and `mask_red2`) to cover the full hue range for red.
   - For green: Creates a single mask for green color.

   **Why**:  
   - This isolates the red and green pixels, making it easier to detect where the traffic lights might be located.

### 5. **Combining Red and Green Masks**
   ```python
   thresh = cv2.bitwise_or(red_thresh, green_thresh)
   ```
   **What it does**:  
   - Combines both red and green masks into a single thresholded image.  
   
   **Why**:  
   - This allows the function to process both red and green regions simultaneously for detection.

### 6. **Finding Contours**
   ```python
   cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   ```
   **What it does**:  
   - Finds the contours (boundaries of objects) in the thresholded image.
   - These contours represent areas where red or green pixels are detected.

   **Why**:  
   - Contours allow us to analyze shapes and sizes of detected regions to filter out irrelevant objects and focus on likely traffic lights.

### 7. **Filtering by Shape and Size**
   ```python
   if area < 3000:
       ...
       aspect_ratio = w / float(h)
       if aspect_ratio >= aspect_ratio_threshold:
           continue
   ```
   **What it does**:  
   - Filters out large regions by checking the area of each contour.
   - Ensures that the aspect ratio is within a threshold to filter out non-circular objects (traffic lights are typically circular).

   **Why**:  
   - Traffic lights are generally small and circular, so this step ensures that only contours with appropriate size and shape are kept for further analysis.

### 8. **Getting the Center of the Contour**
   ```python
   ((cX, cY), radius) = cv2.minEnclosingCircle(c)
   ```
   **What it does**:  
   - Computes the minimum enclosing circle for each valid contour.
   - This gives the center coordinates `(cX, cY)` and the radius of the circle.

   **Why**:  
   - Traffic lights are round, and using an enclosing circle helps us extract the center of each detected traffic light.

### 9. **Classifying the Color**
   ```python
   if np.any(red_thresh[y:y+h, x:x+w]):
       x_red.append(cX)
       y_red.append(cY)
       colors.append(RED)
   elif np.any(green_thresh[y:y+h, x:x+w]):
       x_green.append(cX)
       y_green.append(cY)
       colors.append(GRN)
   ```
   **What it does**:  
   - Checks whether the current region belongs to a red or green light by checking the corresponding mask.
   - Adds the center coordinates `(cX, cY)` to the respective list (red or green).
   - Appends the color to the list of detected colors.

   **Why**:  
   - This allows the function to classify each detected region as either a red or green light and store its location.

### 10. **Returning Results**
   ```python
   return {
       X: x_red + x_green,
       Y: y_red + y_green,
       COLOR: colors,
   }
   ```
   **What it does**:  
   - Returns the x and y coordinates for all detected red and green traffic lights.
   - The `COLOR` list contains the corresponding color for each detected traffic light.

   **Why**:  
   - The final output of the function is used for further processing or visualization of traffic lights in the image.


