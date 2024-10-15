# Import necessary libraries
import os                    # Operating system functions
import cv2                   # OpenCV library for image processing
import numpy as np           # NumPy for numerical operations

# export QT_QPA_PLATFORM=xcb


# Define paths to input image and output directories
img_path = "/home/administrator/Vighnesh-mtech-blood/dataset-master/dataset-master/JPEGImages/BloodImage_00002.jpg"  # Specify the path to your image
save_masks_dir = "/home/administrator/Vighnesh-mtech-blood/Output/masked"       # Specify the path to the output folder for masks
save_cropped_dir = "/home/administrator/Vighnesh-mtech-blood/Output/cropped"  # Specify the path to the output folder for cropped images
save_segmented_dir = "/home/administrator/Vighnesh-mtech-blood/Output/segmented"  # Specify the path to the output folder for segmented images

# Set parameters for auto-saving
# seconds_interval_for_auto_save = 00  # Save the image every specific number of seconds

# Define keys for clearing and toggling drawing mode
# clear_key = "c"  # press "c" for example to clear the drawings
# thick_line_key = "m"  # press "m" for example to use thicker line

# Define the fixed size for cropped images
fixed_image_size = 256  # the desired image size in pixels

# Initialize drawing-related variables
# drawing = False  # True if touch is detected
# mode = True  # True for drawing lines, False for drawing circles
# ix, iy = -1, -1

# Load the input image
img = cv2.imread(img_path)
if img is None:
    print(f"Failed to load image from {img_path}")
    exit()

# Create a copy of the input image for reference
img_copy = img.copy()

# Create a blank black image with the same dimensions as the loaded image
blank_image = np.zeros_like(img)

# Define a callback function for mouse events
def draw(event, x, y, flags, param):
    global ix, iy, drawing, mode, img, blank_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
                cv2.line(blank_image, (ix, iy), (x, y), (0, 0, 255), 2)
                ix, iy = x, y
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(blank_image, (x, y), 5, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.line(blank_image, (ix, iy), (x, y), (0, 0, 255), 2)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(blank_image, (x, y), 5, (0, 0, 255), -1)

# Create an OpenCV window and set the mouse callback function
cv2.namedWindow('Image Editor')
cv2.setMouseCallback('Image Editor', draw)

# Create output directories if they don't exist
if not os.path.exists(save_masks_dir):
    os.makedirs(save_masks_dir)

if not os.path.exists(save_cropped_dir):
    os.makedirs(save_cropped_dir)

if not os.path.exists(save_segmented_dir):
    os.makedirs(save_segmented_dir)

# Set the time interval for auto-saving
save_interval = seconds_interval_for_auto_save
next_save_time = cv2.getTickCount() + save_interval * cv2.getTickFrequency()

# Enter the main loop for image editing
while True:
    cv2.imshow('Image Editor', img)
    key = cv2.waitKey(1) & 0xFF

    # Toggle between drawing modes (lines and circles)
    if key == ord(thick_line_key):
        mode = not mode
    # Exit the application when the 'Esc' key is pressed
    elif key == 27:
        break
    # Clear the drawing when the 'c' key is pressed
    elif key == ord(clear_key):
        img = img_copy.copy()
        blank_image = np.zeros_like(img)

    # Check if it's time to auto-save
    current_time = cv2.getTickCount()
    if current_time >= next_save_time:
        save_filename = os.path.basename(img_path)

        # Find contours on the black image
        gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Define the new bounding box coordinates
            new_x = max(0, (center_x - (int(float(fixed_image_size) / 2))))
            new_y = max(0, (center_y - (int(float(fixed_image_size) / 2))))
            new_w = min((center_x + (int(float(fixed_image_size) / 2))), img.shape[1])
            new_h = min((center_y + (int(float(fixed_image_size) / 2))), img.shape[0])

            # Adjust images for objects close to borders
            left = 0
            top = 0
            right = 0
            bottom = 0

            if (center_x - (int(float(fixed_image_size) / 2))) < 0:
                left = (center_x - (int(float(fixed_image_size) / 2))) * -1
            if (center_y - (int(float(fixed_image_size) / 2))) < 0:
                top = (center_y - (int(float(fixed_image_size) / 2))) * -1
            if (center_x + (int(float(fixed_image_size) / 2))) > img.shape[1]:
                right = (center_x + (int(float(fixed_image_size) / 2))) - img.shape[1]
            if (center_y + (int(float(fixed_image_size) / 2))) > img.shape[0]:
                bottom = (center_y + (int(float(fixed_image_size) / 2))) - img.shape[0]

            # Crop from the bigger image
            Mask_image = blank_image[new_y: new_h, new_x: new_w]
            Cropped_image = img_copy.copy()[new_y: new_h, new_x: new_w]

            # Create a bigger image for the missed part
            completed_mask_area = np.zeros((fixed_image_size, fixed_image_size, 3), dtype=np.uint8)
            completed_cropped_area = np.ones((fixed_image_size, fixed_image_size, 3), dtype=np.uint8) * 255

            # Calculate the coordinates for the missed part
            missed_x = max(0, left)
            missed_y = max(0, top)
            missed_w = fixed_image_size - (left + right)
            missed_h = fixed_image_size - (top + bottom)

            # Add the missed part to the cropped region
            completed_mask_area[missed_y:missed_y + missed_h, missed_x:missed_x + missed_w] = Mask_image
            Mask_image = completed_mask_area

            completed_cropped_area[missed_y:missed_y + missed_h, missed_x:missed_x + missed_w] = Cropped_image
            Cropped_image = completed_cropped_area

            # Fill the contour with white color
            cv2.drawContours(blank_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            # Save the cropped region with an informative filename
            filename = f"{os.path.splitext(save_filename)[0]},{x},{y},{w + x},{h + y}).jpg"
            Mask_save_path = os.path.join(save_masks_dir, filename)
            cv2.imwrite(Mask_save_path, Mask_image)

            Cropped_save_path = os.path.join(save_cropped_dir, filename)
            cv2.imwrite(Cropped_save_path, Cropped_image)

            # Create the segmented image by applying the mask to the cropped region
            Mask_used = cv2.bitwise_not(Mask_image)
            Segmented_image = cv2.addWeighted(Mask_used, 1, Cropped_image, 1, 0)

            # Save the segmented image with the same filename in the save_segmented_dir
            Segmented_save_path = os.path.join(save_segmented_dir, filename)
            cv2.imwrite(Segmented_save_path, Segmented_image)

        # Update the time for the next auto-save
        next_save_time = current_time + save_interval * cv2.getTickFrequency()

# Close all OpenCV windows when the loop ends
cv2.destroyAllWindows()