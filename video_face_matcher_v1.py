
# Author: Rob Hanson
# Required Hardware: Raspberry Pi, Movidius NCS, Adafruit DC and Stepper Motor HAT, LEDs, webcam, 2E10 buggy
#
# Description:  This module is used for face recogniton based navigation of an autonomous vehicle
#               A frame is read from the webcam every ~500 ms and processed.  If a face in the within
#               the webcam's field of view it will be detected, cropped and pre-processed, otherwise  
#               the next frame is read.  Then it is converted to a facial feature vector by FaceNet.
#
#               The feature vector is compared to that of a target face.  If the total error is below
#               FACE_MATCH_THRESHOLD, the faces are deemed to be a match.
#
#               it is based on Movidius' video_face_matcher.py script with several substantial changes:
#                   Face detection to align and crop faces
#                   Proportional only control system for motor control
#                   Overlaying of bounding boxes on detected and recognised faces
#                   LED control for monitoring of the current program status
# 

#--------------------------------------------------------------------
#-------------------------Library Imports----------------------------
#--------------------------------------------------------------------

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
import atexit
import time
from gpiozero import LED

#--------------------------------------------------------------------
#-------------------Hyperparameters & Constants----------------------
#--------------------------------------------------------------------

# Scaling factor for face detection cascade, experiment with this value to see what works for your setup
FD_SCALE_FACTOR = 1.3

# The threshold used for face recognition
# Total squared error between two facial feature vectors must be below this for two faces to be deemed a match
# Lower this if false positives occur, increase if false negatives occur
FACE_MATCH_THRESHOLD = 0.7

#EXAMPLES_BASE_DIR='../../'
IMAGES_DIR = './'

TARGET_IMAGES_DIR = IMAGES_DIR + 'target_images/'

# A single image of the target's face named target.jpg is used as the face to recognise
target_image_filename = TARGET_IMAGES_DIR + 'target.jpg'

GRAPH_FILENAME = "facenet_celeb_ncs.graph"

# OpenCV cascade used for face detection.  A Haar cascade may also be used
#face_cascade = cv2.CascadeClassifier("""./haarcascade_frontalface_default.xml""")
face_cascade = cv2.CascadeClassifier("""./lbpcascade_frontalface_improved.xml""")

# Proportional only motor controller parameters
ERROR_TOLERANCE = 0.01      # Any feedback error below this value will cause buggy to continue straight
MOTOR_OFFSET_SPEED = 25     # Minimum speed difference between left and right motor while error is greater than tolerance
MOTOR_MAX_SPEED = 100

# Name of the opencv window
CV_WINDOW_NAME = "FaceNet"

# Camera parameters, change these to your camera's resolution but bear in mind that
# higher resolution will decrease frame processing time.
CAMERA_INDEX = 0
#REQUEST_CAMERA_WIDTH = 1280
#REQUEST_CAMERA_HEIGHT = 720
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

#--------------------------------------------------------------------
#----------------------Function Definitions--------------------------
#--------------------------------------------------------------------

# Run an inference on the passed image with the FaceNet graph
# Returns 128d facial feature vector
def run_inference(image_to_classify, facenet_graph):
    # Scale, convert, and whiten image
    resized_image = preprocess_image(image_to_classify)

    # Send to NCS
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # Get output from NCS
    output, userobj = facenet_graph.GetResult()
    return output

# Whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# Rescale to network input shape, convert from BGR to RGB and whiten an image 
def preprocess_image(src):
    # Scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    # Convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    # Whiten
    preprocessed_image = whiten_image(preprocessed_image)

    return preprocessed_image

# See if two inference outputs correspond to a matching face (calculates total squared error)
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print('Total Difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        return True
    return False

# Handles cv2.waitkey()
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True

# Initialise and run camera
# Run Haar face detection on each frame
# For each face detected, run an inference to see if it's the right face
# If face is recognised, send control signal to motor 
def run_camera(target_output, target_image_filename, graph):
    # Open camera and set desired resolution
    camera_device = cv2.VideoCapture(CAMERA_INDEX)
    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    # Return camera resolution
    actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('actual camera resolution: ' + str(actual_camera_width) + ' x ' + str(actual_camera_height))

    # If camera can't be opened, return an error message
    if ((camera_device == None) or (not camera_device.isOpened())):
        print ('Could not open camera.  Make sure it is plugged in.')
        print ('Also, if you installed python opencv via pip or pip3 you')
        print ('need to uninstall it and install from source with -D WITH_V4L=ON')
        print ('Use the provided script: install-opencv-from_source.sh')
        return

    # Initialise variables
    frame_count = 0
    cv2.namedWindow(CV_WINDOW_NAME)
    found_match = False

    counter = 0
    
    while True:
        counter += 1
        start = time.time()
        
        # Read video frame from camera
        ret_val, vid_frame = camera_device.read()
        if (not ret_val):
            print("No image from camera, exiting")
            break

        set_LEDs("red")

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(CV_WINDOW_NAME, gray)

        # Returns x, y, w & h (bottom left corner, width and height) of rectangles bounding each face
        face_locations = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FD_SCALE_FACTOR,
            minNeighbors=5,
            minSize=(50, 50),
            maxSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Get image dimensions
        image_height = len(vid_frame)
        image_width = len(vid_frame[0])
        image_centre = int(image_width/2)

        frame_count += 1
        frame_name = 'Camera frame #' + str(frame_count)

        num_faces = 0
        
        # For each face detected
        for (x, y, w, h) in face_locations:
            # Increase the size of the bounding box
            x = int(max(0,x-0.5*w))
            w = 2*w
            y = int(max(0, y-0.5*h))
            h = 2*h
            set_LEDs("yellow")
            num_faces += 1

            # Run an inference for each face detected
            face_image = vid_frame[y:y+h, x:x+w]
            test_output = run_inference(face_image, graph)

            # Overlay image and update motors
            if (face_match(target_output, test_output)):
                set_LEDs("green")
                
                # Determine position of the centre of the face in the frame (as a magnitude left/right relative to the centre of the image)
                face_centre = [int((y+y+h)/2), int((x+x+w)/2)]
                direction = (face_centre[1] - image_centre)/image_centre

                # Print result
                print('PASS! ', frame_name, ", face ", num_faces, ' matches ', target_image_filename)
                
                # Overlay bounding box and add text of magnitude left or right
                cv2.rectangle(vid_frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(vid_frame,  str(direction), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

                print (direction)

                update_motor_speed(direction)
     
            else:
                # Face was not a match
                found_match = False
                print('FAIL! ', frame_name, ", face ", num_faces, ' does not match ', target_image_filename)
                cv2.rectangle(vid_frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

        if num_faces == 0:
            print ("No face detected")
            # Turn off all motors
            rear_right.run(Adafruit_MotorHAT.RELEASE)
            rear_left.run(Adafruit_MotorHAT.RELEASE)
            front_right.run(Adafruit_MotorHAT.RELEASE)
            front_left.run(Adafruit_MotorHAT.RELEASE)
            
        # check if the window is visible, this means the user hasn't closed
        # the window via the X button
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            print('window closed')
            break

        # display the results and wait for user to hit a key
        cv2.imshow(CV_WINDOW_NAME, vid_frame)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed Q')
                return

        end = time.time()
        print ("Time taken to process frame: ", str(end-start)[:5], "s\n")

# Takes float input (-1 [full left] to 1 [full right]) and updates the H bridge accordingly
# Also uses area of bounding box / height of bounding box to determine how fast it should
def update_motor_speed(error):
    set_speed = int(MOTOR_MAX_SPEED*0.5)
    
    if error > ERROR_TOLERANCE:
        # Turn right
        left_speed = set_speed
        right_speed = int(max(1 - 2 * error, 0) * set_speed)

    elif error < ERROR_TOLERANCE:
        # Turn left
        right_speed = set_speed
        left_speed = int(max(1 + 2 * error, 0) * set_speed)

    else:
        # Go straight
        left_speed = set_speed
        right_speed = set_speed

    # Update motor state
    front_right.setSpeed(right_speed)
    rear_right.setSpeed(right_speed)
    front_left.setSpeed(left_speed)
    rear_left.setSpeed(left_speed)

    #print ("Right speed: ", right_speed)
    #print ("Left speed: ", left_speed)
    rear_right.run(Adafruit_MotorHAT.FORWARD);
    rear_left.run(Adafruit_MotorHAT.FORWARD);
    front_right.run(Adafruit_MotorHAT.FORWARD);
    front_left.run(Adafruit_MotorHAT.FORWARD);

# Turns on the LED colour passed to the function, turns off all others
def set_LEDs(color):
    # Turn off all lights
    green_LED.off()
    yellow_LED.off()
    red_LED.off()

    # Update LED
    if color == "green":
        green_LED.on()
        #print ("Green")
    elif color == "yellow":
        yellow_LED.on()
        #print ("Yellow")
    elif color == "red":
        red_LED.on()
        #print ("Red")
    else:
        print ("LEDs powered down")


#--------------------------------------------------------------------
#-----------------------------IO Setup-------------------------------
#--------------------------------------------------------------------

# create a default object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT(addr=0x60)

# Setup each motor
rear_right = mh.getMotor(1)     # 
rear_left = mh.getMotor(2)      # 
front_right = mh.getMotor(3)    # 
front_left = mh.getMotor(4)     #

# GPIO PINS
# SDA & SCL in use for motors' PWM
# Using pin 6 as GND for all 3 LEDs
green_LED = LED(12)     # Physical pin 32
yellow_LED = LED(16)    # Physical pin 36
red_LED = LED(21)       # Physical pin 40

#--------------------------------------------------------------------
#----------------------------NCS Setup-------------------------------
#--------------------------------------------------------------------

# Check if NCS is connected
devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print('No NCS devices found')
    quit()

# Pick the first stick to run the network
device = mvnc.Device(devices[0])

# Open the NCS
device.OpenDevice()

# The graph file that was created with the ncsdk compiler
graph_file_name = GRAPH_FILENAME

# read in the graph file to memory buffer
with open(graph_file_name, mode='rb') as f:
    graph_in_memory = f.read()

# create the NCAPI graph instance from the memory buffer containing the graph file.
graph = device.AllocateGraph(graph_in_memory)
del graph_in_memory
del f

#--------------------------------------------------------------------
#-------------Generate feature vector for target face----------------
#--------------------------------------------------------------------

# Load training image
target_image = cv2.imread(target_image_filename)

# Convert image to grayscale for face detection
gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Returns a tuple containing numpy.int32s representing rectangles bounding each face
face_locations = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# Run inference on target image
if len(face_locations) == 1:
    # Unpack face locations
    for (x, y, w, h) in face_locations:
        # Increase the boundary around each face
        x = int(max(0,x-0.5*w))
        w = 2*w
        y = int(max(0, y-0.5*h))
        h = 2*h
        
        # Run an inference for face
        face_image = target_image[y:y+h, x:x+w]
        target_output = run_inference(face_image, graph)
        print (target_output)
                
        # Overlay bounding box
        cv2.rectangle(target_image, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(target_image, ("Training face"), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
else:
    print (len(face_locations), "faces detected in training image.")
    print ("Try retaking training image")
    quit()

print ("Press any key to continue...")
cv2.imshow("Training image", target_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

run_camera(target_output, target_image_filename, graph)

# Clean up the graph and the device
graph.DeallocateGraph()
device.CloseDevice()

# Turn off all motors
rear_right.run(Adafruit_MotorHAT.RELEASE)
rear_left.run(Adafruit_MotorHAT.RELEASE)
front_right.run(Adafruit_MotorHAT.RELEASE)
front_left.run(Adafruit_MotorHAT.RELEASE)
print ("Motors shutdown")

set_LEDs("")

# Close all windows
cv2.destroyAllWindows()

