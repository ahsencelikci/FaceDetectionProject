# Face Recognition System

## Overview
This project implements a face recognition system utilizing deep learning techniques on the AT&T (ORL) Face Database. The system employs Convolutional Neural Networks (CNN) to identify individuals based on their facial images across various expressions.

## Dataset
The AT&T (ORL) Face Database contains:
- **40 subjects** (s1 to s40).
- Each subject has **10 images** depicting different expressions (e.g., smiling, wearing glasses).
- The images are in **PGM format** and are organized into folders by subject.

## Features
- **Face Recognition**: Identifies individuals from facial images using a trained model.
- **Deep Learning**: Utilizes CNN for effective feature extraction and classification.
- **Evaluation Metrics**: Assesses model performance using accuracy and confusion matrices.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required libraries:
  - `numpy`
  - `opencv-python`
  - `tensorflow`
  - `keras`
  - `scikit-learn`
### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/FaceDetectionProject.git
   cd FaceDetectionProject
   
2. **Install required packages: Create a requirements.txt file with the following content**:
  - `numpy`
  - `opencv-python`
  - `tensorflow`
  - `keras`
  - `scikit-learn`

Then run:
 ```bash
    pip install -r requirements.txt
```

## Directory Structure
 ```bash
    FaceDetectionProject/
│
├── data/                    # Contains the dataset
│   ├── s1/
│   ├── s2/
│   ├── ...
│   └── s40/
│
├── model.py                 # Contains the model architecture
├── preprocess.py            # Contains data preprocessing functions
├── train.py                 # Script for training the model
└── README.md                # Project documentation

```
## Usage
1. Preprocess the dataset:

    - Run preprocess.py to prepare images for training.

2. Train the model:

    - Execute train.py to start training the CNN model.

3. Evaluate the model:

    - The training script includes evaluation metrics to assess performance on the test dataset.
  

## Results
The model will output the accuracy of face recognition and display a confusion matrix, helping to analyze its effectiveness in identifying different subjects.
