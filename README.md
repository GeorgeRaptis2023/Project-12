# Project-12
To run examplev2.py the modules cv2,face-recognition,sklearn,tensorflow must be installed following the official instructions 
for [sklearn](https://scikit-learn.org/stable/install.html),
for [face-recognition]( https://pypi.org/project/face-recognition/),
for [cv2](https://pypi.org/project/opencv-python/).
for [tensorflow](https://www.tensorflow.org/install).
We are using scikit-learn "1.3.2",Pillow "10.1.0",opencv-python "4.8.1.78",face-recognition "1.3.0", tensorflow "2.10.1".
There must also be a folder named 'train' with folders inside it named after each person with photos(jpg) of that person for the k-nearest neighbor classifier and our model.
In lines 143 and 144 the path must be changed depending on where the folder train is located.
The model can become more accurate if more images are added in the folders of train.
