# Malaria Detection using Deep Learning

## Overview

This repository contains a Jupyter Notebook (`Malaria_Diagonisis.ipynb`) that demonstrates the use of deep learning techniques for the detection of malaria in cell images. The project leverages TensorFlow and TensorFlow Datasets to build, train, and evaluate a convolutional neural network (CNN) model for binary classification of malaria-infected and uninfected cells.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used in this project is the **Malaria Dataset** from TensorFlow Datasets. It contains a total of 27,558 cell images, with equal instances of parasitized and uninfected cells. The images are derived from thin blood smear slide images of segmented cells.

### Dataset Details:
- **Name**: Malaria
- **Size**: 337.08 MiB
- **Features**:
  - `image`: RGB images of varying dimensions.
  - `label`: Binary labels (0 for uninfected, 1 for parasitized).
- **Splits**:
  - Train: 27,558 images

## Dependencies

To run the notebook, you need the following Python libraries:

- TensorFlow
- NumPy
- Matplotlib
- TensorFlow Datasets
- Albumentations
- OpenCV
- Scikit-learn
- Seaborn
- WandB (Weights & Biases)

You can install the required libraries using the following command:

```bash
pip install tensorflow numpy matplotlib tensorflow-datasets albumentations opencv-python scikit-learn seaborn wandb
```

## Installation


Clone the repository:

```bash
git clone https://github.com/your-username/malaria-detection.git
cd malaria-detection
```
Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage
The notebook is structured to guide you through the entire process of building and evaluating a deep learning model for malaria detection:

Data Loading and Preprocessing:
1. Load the Malaria dataset using TensorFlow Datasets.
2. Split the dataset into training, validation, and test sets.
3. Apply data augmentation techniques to improve model generalization.

Model Architecture:
1. Define a CNN model using TensorFlow's Keras API.
2. Include layers such as Conv2D, MaxPool2D, Dropout, and Dense layers.
3. Use Binary Crossentropy as the loss function and Adam as the optimizer.

Training:
1. Train the model on the training dataset.
2. Use early stopping and learning rate scheduling to prevent overfitting.
3. Log training metrics using WandB for visualization.

Evaluation:
1. Evaluate the model on the validation and test datasets.
2. Compute metrics such as accuracy, precision, recall, and F1-score.
3. Visualize the confusion matrix and ROC curve.

Results:
1. Analyze the model's performance and discuss potential improvements.
2. Save the trained model for future use.


## Model Architecture
The model architecture consists of the following layers:
1. Input Layer: Accepts images of varying dimensions.
2. Convolutional Layers: Multiple Conv2D layers with ReLU activation.
3. MaxPooling Layers: MaxPool2D layers to reduce spatial dimensions.
4. Dropout Layers: Dropout layers to prevent overfitting.
5. Flatten Layer: Flattens the output from the convolutional layers.
6. Dense Layers: Fully connected layers with ReLU activation.
7. Output Layer: Dense layer with sigmoid activation for binary classification.


## Training
The model is trained using the Adam optimizer with a learning rate scheduler. Early stopping is implemented to halt training if the validation loss does not improve for a specified number of epochs.

Hyperparameters:
1. Batch Size: 32
2. Epochs: 50
3. Learning Rate: 0.001
4. Early Stopping: Patience of 5 epochs

## Evaluation
The model's performance is evaluated using the following metrics:
1. Accuracy: Percentage of correctly classified images.
2. Precision: Proportion of true positives among predicted positives.
3. Recall: Proportion of true positives among actual positives.
4. F1-Score: Harmonic mean of precision and recall.

## Visualization:
- Confusion Matrix: Visualizes the model's predictions.
- ROC Curve: Plots the true positive rate against the false positive rate.

## Results
The model achieves high accuracy and precision in classifying malaria-infected and uninfected cells. Detailed results, including metrics and visualizations, are available in the notebook.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.
- Fork the repository.
- Create a new branch (git checkout -b feature/YourFeatureName).
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature/YourFeatureName).
- Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to explore the notebook and experiment with different model architectures and hyperparameters to further improve the performance of the malaria detection model. If you have any questions or need further assistance, please don't hesitate to reach out.
