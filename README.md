# Pose-Recognition
Applied Machine Learning (EEET-520) final project - pose recognition team

There are two main files in this repo: 

  CreatePoseTrainingDataCsv.py takes in images from the poses_in folder and creates a training set file called poses_out.csv. Inside the poses_in folder, the name of each subfolder dictates the class of the images     inside.

  TrainAndRunPoseDetectionSVM.py accepts poses_out.csv as training data to train an SVM and then runs that SVM on webcam images.

Use Instructions:

  If you simply want to train/run the SVM, download TrainAndRunPoseDetectionSVM.py and poses_out.csv. These will need to be in the same directory when you run the Python file.

  If you want to add more training data to the training set download CreatePosesTrainingDataCsv.py, TrainAndRunPoseDetectionSVM.py, and the entire poses_in directory. With pose the poses_in directory in the same      directory as CreatePoseTrainingDataCsv.py, add your images to poses_in following the structure that already exists then run that Python file. This will generate a new poses_out.csv which can then be used to train 
  the SVM as mentioned above. If you add significant high-quality training images make sure to push both the new images and poses_out.csv to the repo.
