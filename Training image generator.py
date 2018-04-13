# Takes an image with a webcam and saves to ./target_images/target.jpg

import cv2
import time

#REQUEST_CAMERA_WIDTH
#REQUEST_CAMERA_HEIGHT
 
warmup_frames = 30
 
# Initialise camera
camera_device = cv2.VideoCapture(0)

actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
print ('actual camera resolution: ' + str(actual_camera_width) + ' x ' + str(actual_camera_height))
 
# Captures a single image from the camera
def get_image():
    ret_val, image = camera_device.read()
    return image

# Take several images to allow the camera to auto-expose and focus
print ("Warming up...")
for i in range(warmup_frames):
    # Get an image from the camera
    camera_capture = get_image()

    # Give the user a warning
    if i == warmup_frames - 5:
        print ("Say cheese")

# Get the final image
camera_capture = get_image()

# Display the image
print ("Press any key to save")
cv2.imshow("Training image", camera_capture)
cv2.waitKey(0)

# Save the image
file = "target_images/target.jpg"
cv2.imwrite(file, camera_capture)
print("Image saved")

cv2.destroyAllWindows()

# Deconstructor
del(camera_device)
