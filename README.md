# Final-Year-Face-Recognition-Project

This page details my final year project on neural network based face recognition for autonomous vehicle navigation.

Contained in the repository are the final report, source code, testing results and demonstration of a deep neural network based face recognition system deployed on an autonomous vehicle (AV) using a Raspberry Pi (RPi), Intel Movidius NCS and a webcam.

video_face_matcher_v1.py demonstrates AV navigation towards a target face when deployed on a RPi.  It uses a FaceNet implementation to create a Euclidean embedding of each face in a webcam's FoV, where the squared distance between these embeddings corresponds directly to facial similarity.  If this distance is below a threshold, the faces are deemed to be a match and the AVs direction is updated.  An OpenCV cascade classifier is used for face detection and alignment.

LFW test.py runs a test on the LFW dataset.  It requires only an NCS and Ubuntu machine.  To run this test, download the LFW dataset: http://vis-www.cs.umass.edu/lfw/ and extract to ./lfw/

