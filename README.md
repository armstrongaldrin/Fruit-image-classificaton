# Fruit-image-classificaton

https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition data-set 



Fruit Image Classification using Convolutional Neural Network (CNN)
This project aims to classify images of various fruits using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The model is trained and validated using a dataset of fruit images, achieving accuracy and generating visualizations of the training process.

Project Overview
The project includes the following steps:

Data Preprocessing:

Images are loaded from directories using TensorFlow's image_dataset_from_directory function.
The dataset is split into training and validation sets.
Images are resized to 64x64 pixels and batched for model input.
Model Architecture:

A Sequential CNN model is built with the following layers:
Convolutional layers with ReLU activation
MaxPooling layers for downsampling
Flattening layer to convert the 2D matrix to a vector
Dense layers with ReLU activation
Dropout layer to prevent overfitting
Output layer with softmax activation for multi-class classification
Model Compilation and Training:

The model is compiled using the Adam optimizer and categorical cross-entropy loss.
The training process is monitored using the validation dataset.
The model is trained over 32 epochs.
Model Evaluation:

The final accuracy on the validation set is printed.
Training and validation accuracy are visualized using matplotlib.
Model Saving:

The trained model is saved as trained_model.h5.
The training history is saved as a JSON file for further analysis.
Prerequisites
To run this project, you will need:

Python 3.x
TensorFlow
NumPy
Matplotlib
Google Colab (optional, for running on cloud)
