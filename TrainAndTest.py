import cv2
import csv
import os
import sys
import matplotlib.pyplot as plt
# import tqdm
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Train the SVM
X = []
y = []

# Read the values in the csv training file
with open('poses_out.csv', newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        X.append(row[2:])
        y.append(row[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.44)

svms = {}
i = 0.01
for _ in range(10):
  svm_clf = SVC(kernel="rbf", C=i)
  svm_clf.fit(X, y)
  svms[i] = svm_clf
  i = i*10

from sklearn.metrics import accuracy_score


i = 0.01
for _ in range(10):
  y_pred = svms[i].predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'{i} : Accuracy:', accuracy)
  i = i*10
# import pickle
# with open('model.pkl','wb') as f:
#     pickle.dump(svm_clf,f)

class_names = ['GOLf', 'NO_POSE', 'START', 'STOP', 'WAVE_RIGHT_HAND', 'BICEP_CURL']
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        svms[10],
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()