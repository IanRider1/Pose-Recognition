import pickle
import cv2
import csv
import os
import sys
# import tqdm
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

X = []
y = []

def get_mode(list1):
    # Create a dictionary to store the frequency of each element
    freq = {}
    for element in list1:
        if element in freq:
            freq[element] += 1
        else:
            freq[element] = 1

    # Find the element with the highest frequency
    max_freq = 0
    mode = None
    for element, frequency in freq.items():
        if frequency > max_freq:
            max_freq = frequency
            mode = element

    return mode

with open('model.pkl', 'rb') as f:
    svm_clf = pickle.load(f)

testData = 'pose_webcam.csv'

count = 0
last_ten = [None for _ in range(10)]
with open(testData, 'w') as csv_out_file:  # Open csv here
  csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL) # Writer

  cap = cv2.VideoCapture(0) # Video Capture
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      landmarks = results.pose_landmarks

      if landmarks is not None:  # Makes sure to keep program running in case there is no one in frame
        assert len(landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(landmarks.landmark))
        # prints the len of the list landmarks
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Write to csv file of landmarks
        landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark]

        # Map pose landmarks from [0, 1] range to absolute coordinates to get
        # correct aspect ratio.
        frame_height, frame_width = image.shape[:2]
        landmarks *= np.array([frame_width, frame_height, frame_width])

        # Write to csv
        landmarks = np.around(landmarks, 5).flatten().astype(str).tolist()  # Convert to a type the csv can read/write
        csv_out_writer.writerow(landmarks)
      
        # Read and predict class
        with open('pose_webcam.csv', newline='\n') as csvfile:
          spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
          for row in spamreader:
              X.append(row)
              
        if (len(X) > 1):
            X_new = [X[-1]] # Reads last frame?
            last_ten[count%10] = svm_clf.predict(X_new)[0]
            print(get_mode(last_ten))
            count += 1
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
      if cv2.waitKey(1) == ord('q'):
        csv_out_file.close()  
        break
cap.release()