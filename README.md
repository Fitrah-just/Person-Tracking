# Introduce our team project
- Hendra
- Fatturahman
- Yogi
- Harisson
- Dani
- Fitrah
- Nadya
- Satriaji
- Tri Wahyu
  

# Person-Tracking using YOLO & Faster Rcnn

Person tracking is a vital aspect of computer vision applications, playing a pivotal role in surveillance, security, and various human-computer interaction systems. Leveraging state-of-the-art object detection algorithms is crucial for achieving accurate and real-time person tracking. Two such algorithms are YOLO (You Only Look Once) and Faster R-CNN (Region-based Convolutional Neural Network), each incorporating different strategies for object detection. Furthermore, the choice of backbone architectures, such as ResNet50, MobileNet, and GoogleNet, significantly influences the performance of these algorithms in person tracking scenarios.

1. YOLO (You Only Look Once):
YOLO is a real-time object detection algorithm that stands out for its efficiency and speed. It operates by dividing the input image into a grid and simultaneously predicts bounding boxes and class probabilities for each grid cell. YOLO's single-shot approach makes it particularly well-suited for person tracking applications where swift and accurate detection is essential.

2. Faster R-CNN:
Faster R-CNN, on the other hand, is a two-stage object detection framework. It introduces a Region Proposal Network (RPN) to generate potential bounding box proposals, followed by the prediction of final bounding boxes and class probabilities. While it may have a slightly higher computational cost, Faster R-CNN excels in scenarios where precision and accuracy are paramount in person tracking.

3. Backbone Architectures:

- ResNet50:
ResNet50, derived from the Residual Network architecture, is characterized by its deep structure with residual connections. In the context of person tracking, ResNet50 proves advantageous in capturing intricate features and patterns from the input images. This makes it particularly effective in scenarios where detailed information is crucial for accurate tracking.

- MobileNet:
MobileNet is renowned for its lightweight and efficient design, making it suitable for deployment in resource-constrained environments, including mobile and edge devices. The depthwise separable convolutions in MobileNet contribute to reduced computational complexity while maintaining accuracy, making it an excellent choice for real-time person tracking in such settings.

- GoogleNet (Inception):
GoogleNet, also known as Inception, introduced inception modules that facilitate the capture of features at multiple scales. This backbone architecture excels in handling diverse and multi-scale information. In person tracking scenarios, GoogleNet is beneficial when dealing with varied environments and scales, providing robust tracking capabilities.

In summary, the combination of YOLO and Faster R-CNN algorithms with diverse backbone architectures like ResNet50, MobileNet, and GoogleNet provides a versatile toolkit for person tracking applications. The choice of algorithm and backbone depends on the specific requirements of the application, considering factors such as speed, precision, and resource constraints. This combination ensures adaptability to a wide range of scenarios, contributing to the effectiveness of person tracking systems.

![image](https://github.com/Fitrah-just/Person-Tracking/assets/84637046/99fc2f60-a2c4-4ebd-9ae5-e970d88db676)


## Dataset
For our dataset, we utilized the one provided by FiftyOne, selecting only 5000 samples consisting exclusively of the "person" class for experimentation. 
Out of these **5000 samples**, we divided them into **4000 for training**, **500 for validation**, and another **500 for testing purposes**.

## Import your dataset using fiftyone
```
!pip install fiftyone
!pip install fiftyone-db-ubuntu2204

!wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py
!wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py
!wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py
!wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py
!wget https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py
```

Use FiftyOne to Import the COCO Dataset
You can use a Python script to import the COCO 2017 dataset into FiftyOne. 
Here is an example code to import the dataset:
```
import fiftyone as fo
import fiftyone.zoo as foz

# Load the COCO-2017 dataset
# This will download it from the FiftyOne Dataset Zoo if necessary
dataset = foz.load_zoo_dataset("coco-2017", split="train", label_types=["detections"], classes=["person"], max_samples=4000)
dataset_test = foz.load_zoo_dataset("coco-2017", split="validation", label_types=["detections"], classes=["person"], max_samples=500)

# Print summary information about the view
print(dataset)
```

## Dependencies

Following Python libraries installed:

* Basic Libraries: [NumPy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/), [Pandas](https://pandas.pydata.org/), [Sklearn](https://scikit-learn.org/)
* Deep-learning Frameworks: [Keras](https://keras.io/), [PyTorch](https://pytorch.org/), [Ultralytics](https://www.ultralytics.com/)

📨 That's all, for any discussion kindly contact me here: ramadhanfitrah2@gmail.com

