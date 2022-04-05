# Facial Emotion Recognition
This package supports 1) **Training**, 2) **Inferring**, 3) **real-time webcam** for Facial Emotion Recognition.

## 0. Dependencies
Install [PyTorch](https://pytorch.org/get-started/locally/).\
Install [OpenCV](https://pypi.org/project/opencv-python/).\
Install [sklearn](https://anaconda.org/anaconda/scikit-learn).\
Install PIL, matplotlib, numpy, etc.

## 1. Extract (crop) and align face from an image
```python extract_face.py```

DIR_IN: raw image path (before face extraction). The code expects one-level hierarchy under DIR_IN (folders under DIR_IN).\
DIR_OUT: face data path (after face extraction) with same one-level hierarchy.\
NUM_IMAGES: number of images under each folder under DIR_IN to extract faces; 1) positive: from the first, 2) negative: from the last, 3) 0: all images

## 2. Train
```python train.py```

Modify settings via argument parsing.\
This code results in ```f'{args.identifier}_model_best.pth.tar'```

## 3. Infer
```python infer.py```
Use ```f'{args.identifier}_model_best.pth.tar'``` from ```python train.py``` as --model.

The user has three options for ```--image```:
1) If --image is ```None```, the code test all images under ```--data``` path and save the results under ```--results```
2) If --image is ```fault_finder```, the code makes csv file that includes how/what images were inferred incorrectly.
3) If --image is a specific path to an image, the code prints the probability and the class with the maximum probability on console.


## 4. Webcam
```python webcam.py```

model(```*.pth.tar```) must exists at ```--model```.\
Use ```--gpu, --fps``` for your convenience.\
```--resize-h``` is 0 by default, which does not resize the input from the real-time webcam at all.
