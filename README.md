# Facial Landmarks Detection using Dlib and OpenCV

Facial landmarks are used to localize and represent salient regions of the face, such as: eyes, eyebrows, nose, mouth, jawline. 

Facial landmarks have been successfully applied to face alignment, head pose estimation, face swapping, blink detection and much more.

Given an input image (and normally an ROI that specifies the object of interest), a shape predictor attempts to localize key points of interest along the shape.

In the context of facial landmarks, our goal is detect important facial structures on the face using shape prediction methods.

Detecting facial landmarks is therefore a two step process:

Step #1: Localize the face in the image using dlib's pre-trained HOG + Linear SVM face detector .

Step #2: Detect the key facial structures on the face ROI using dlib's shape predictor. 

## Dependencies ##

* Dlib
* OpenCV
* Numpy
