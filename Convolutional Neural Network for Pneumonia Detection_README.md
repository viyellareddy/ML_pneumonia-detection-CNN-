

# Convolutional Neural Network for Pneumonia Detection

## Introduction

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to detect pneumonia from chest X-ray images. The model is trained on a dataset of chest X-ray images and evaluates its performance on a separate test set.

## Dataset

- **Dataset Name**: Chest X-ray Images (Pneumonia)
- **Source**: The dataset is publicly available and can be found on Kaggle [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- **Description**: The dataset contains 5,863 X-ray images with two categories: 'Pneumonia' and 'Normal'. The images are divided into training, validation, and test sets.

## Requirements

To run the code, you need to have the following libraries installed:

```bash
pip install tensorflow matplotlib seaborn
```

## Model Architecture

The CNN model used in this project has the following architecture:
- Input Layer: 224x224x3
- Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer 1: 2x2 pool size
- Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer 2: 2x2 pool size
- Convolutional Layer 3: 128 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer 3: 2x2 pool size
- Convolutional Layer 4: 128 filters, 3x3 kernel, ReLU activation
- Max Pooling Layer 4: 2x2 pool size
- Dense Layer 1: 512 units, ReLU activation
- Output Layer: 1 unit, Sigmoid activation

## Results

The model was trained for 10 epochs with a batch size of 32. The training and validation accuracy and loss were plotted to visualize the performance.

### Test Accuracy

The model achieved a test accuracy of approximately [insert accuracy here].

### Confusion Matrix

The confusion matrix was calculated to evaluate the performance of the model in distinguishing between pneumonia and normal cases.

### ROC Curve

The ROC curve and AUC score were plotted to assess the model's performance.

### Precision-Recall Curve

The precision-recall curve and average precision score were plotted to evaluate the precision and recall of the model.

## Conclusion

In this project, a CNN was implemented to detect pneumonia from chest X-ray images. The model demonstrated good performance in distinguishing between pneumonia and normal cases. This project highlights the potential of CNNs in medical image classification tasks and provides a basis for further improvements and applications in the healthcare domain.

