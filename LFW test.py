#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


#from PIL import Image, ImageDraw

from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os
import time

# Set larger for further away faces
FD_SCALE_FACTOR = 1.3

EXAMPLES_BASE_DIR='../../'
IMAGES_DIR = './'

VALIDATED_IMAGES_DIR = IMAGES_DIR + 'validated_images/'

# Takes only a single validated image named valid.jpg
validated_image_filename = VALIDATED_IMAGES_DIR + 'valid.jpg'

GRAPH_FILENAME = "facenet_celeb_ncs.graph"
faceCascade = cv2.CascadeClassifier("""./haarcascade_frontalface_default.xml""")

# name of the opencv window
CV_WINDOW_NAME = "FaceNet"

# Camera parameters
CAMERA_INDEX = 0
#REQUEST_CAMERA_WIDTH = 1280
#REQUEST_CAMERA_HEIGHT = 720
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
FACE_MATCH_THRESHOLD = 0.7

# Run an inference on the passed image
# FaceNet graph will be used to perform inference
# Returns 128d facial feature vector
def run_inference(image_to_classify, facenet_graph):
    # Scale, convert, and whiten image
    resized_image = preprocess_image(image_to_classify)

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(resized_image.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()
    return output

# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

# determine if two images are of matching faces based on the
# the network output for both images.
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print("Total squared error:", str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
        # the total difference between the two is under the threshold so
        # the faces match.
        return True

    # differences between faces was over the threshold above so
    # they didn't match.
    return False

# handles key presses
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True

# Runs FD and FR on all pictures in the LFW folder
def run_test(valid_output, graph):
    matches = 0
    match_list = []
    miss_list = []
    no_face_list = []

    test_image_list = []

    print ("Loading image file names")

    # Add get file path of each test image
    for file in os.listdir("./lfw"):
        
        for image in os.listdir("./lfw/" + file):
            #print(image)
            #print (file)
            test_image_list.append("./lfw/" + file + "/" + image)
    
    num_test_images = len(test_image_list)

    if num_test_images < 10000:
        print ("Not enough test images found, please ensure at least 10,000 images are in the ./lfw directory")
        break
    else
        print ("Image file names loaded")

    # Load first image
    test_image = cv2.imread(test_image_list[0])

    # Display image
    print ("Displaying first image, press any key to continue...")
    print (num_test_images, "images to be tested")

    cv2.imshow(CV_WINDOW_NAME, test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Open a .txt file to store results
    test_txt = open("test_results.txt", "w")
    test_txt.write("Image no., Image file path, is target, result, time taken\n")

    for i in range(0, num_test_images - 1):
        print ("Processing image no. " + str(i+1))
        
        # Save image number and file path
        test_txt.write(str(i+1))
        test_txt.write(", ")
        test_txt.write(test_image_list[i])
        test_txt.write(", ")

        # Check if test face and trained face belong to the same person
        if test_image_list[i][6:18] == "Colin_Powell":
            test_txt.write("yes, ")
        else:
            test_txt.write("no, ")
        
        start = time.time()
        
        # Read next image
        test_image = cv2.imread(test_image_list[i])
            
        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

        # Returns a tuple containing numpy.int32s representing rectangles bounding each face
        face_locations = faceCascade.detectMultiScale(
            gray,
            scaleFactor=FD_SCALE_FACTOR,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        image_height = len(test_image)
        image_width = len(test_image[0])
        image_centre = int(image_width/2)
        num_faces = 0
        
        # For each face detected
        for (x, y, w, h) in face_locations:
            num_faces += 1

            # Run an inference for each face detected
            face_image = test_image[y:y+h, x:x+w]
            test_output = run_inference(face_image, graph)

            # Print output
            if (face_match(valid_output, test_output)):
                
                # Determine position of face in image (left/right, magnitude)
                face_centre = [int((y+y+h)/2), int((x+x+w)/2)]
                
                if (face_centre[1]/image_width > 0.5):
                    direction = ["right", (face_centre[1] - image_centre)/image_centre]
                else:
                    direction = ["left", (image_centre - face_centre[1])/image_centre]

                # Print result
                print('PASS! ', test_image_list[i], ' matches ', validated_image_filename)
                
                # Overlay bounding box, direction and magnitude
                cv2.rectangle(test_image, (x,y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(test_image, (direction[0] + " " + str(direction[1])), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
                #match_list.append(test_image_list[i])
                test_txt.write("faces match, ")
                
            else:
                found_match = False
                print('FAIL! ', test_image_list[i], ' does not match ', validated_image_filename)
                cv2.rectangle(test_image, (x,y), (x+w, y+h), (0, 0, 255), 2)
                #miss_list.append(test_image_list[i])
                test_txt.write("faces don't match, ")

        if num_faces == 0:
            no_face_list.append(test_image_list[i])
            test_txt.write("no face detected, ")

        # display the results and wait for user to hit a key
        cv2.imshow(CV_WINDOW_NAME, test_image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed Q')
                break

        end = time.time()
        test_txt.write(str(end-start)[:5])
        test_txt.write("\n")
        print ("Time taken to process frame: ", str(end-start)[:5], "s")

    test_txt.close()

# This function is called from the entry point to do
# all the work of the program
def main():

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
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

    # Open test images
    #tar0 = tarfile.open('mytar.tar')
    #np.asarray(bytearray(tar_extractfl.read()), dtype=np.uint8)

    # Detect face in training image (must convert to grayscale)
    validated_image = cv2.imread(validated_image_filename)
    gray = cv2.cvtColor(validated_image, cv2.COLOR_BGR2GRAY)

    face_location = faceCascade.detectMultiScale(
        gray,
        scaleFactor=FD_SCALE_FACTOR,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print (face_location)
    
    # Run inference for face detected
    for (x, y, w, h) in face_location:
        print ("found face")
        # Run an inference on the training face
        face_image = validated_image[y:y+h, x:x+w]
        valid_output = run_inference(face_image, graph)
        cv2.rectangle(validated_image, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(CV_WINDOW_NAME, validated_image)
    cv2.waitKey(0)
    #valid_output = run_inference(validated_image, graph)

    run_test(valid_output, graph)

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
