# GarDoc: Plant Disease Classification System

GarDoc is a machine learning-based system for identifying plant diseases from images. This tool uses a fine-tuned ResNet50 model to classify plant species and their health conditions.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Model Architecture](#model-architecture)
- [Dataset Requirements](#dataset-requirements)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (venv or conda)

## Installation

1. Clone the repository or download the source code:

```bash
git clone <repository-url>
cd gardoc
```
2. Create and activate a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
 ```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
 ```

## Project Structure
```plaintext
gardoc/
├── dataset/                  # Training dataset directory
│   ├── plant1_healthy/       # Class folders with images
│   ├── plant1_disease1/
│   └── ...
├── train_model.py            # Script for training the model
├── predict.py                # Script for making predictions
├── requirements.txt          # Required Python packages
├── plant_classification_model.h5    # Trained model (after training)
├── class_mapping.txt         # Class mapping file (after training)
├── training_history.png      # Training visualization (after training)
├── confusion_matrix.png      # Model evaluation (after training)
└── README.md                 # This file
 ```
```

## Training the Model
1. Prepare your dataset in the following structure:
   
   ```plaintext
   dataset/
   ├── plant1_healthy/        # Images of healthy plant1
   ├── plant1_disease1/       # Images of plant1 with disease1
   ├── plant2_healthy/        # Images of healthy plant2
   └── ...
    ```
   ```
   
   Each folder name should follow the format: plantname_condition
2. Run the training script:
   
   ```bash
   python train_model.py
    ```
3. The training process will:
   
   - Split the dataset into training and validation sets
   - Train the model in two phases (top layers, then fine-tuning)
   - Save the best model as plant_classification_model.h5
   - Generate training visualizations and evaluation metrics
   - Create a class mapping file ( class_mapping.txt )
## Making Predictions
1. Ensure you have a trained model ( plant_classification_model.h5 ) and class mapping file ( class_mapping.txt ) in your project directory.
2. Run the prediction script:
   
   ```bash
   python predict.py
    ```
3. When prompted, enter the path to the image you want to analyze.
4. The script will display:
   
   - The original image
   - The predicted plant name and condition
   - Confidence score
   - Top 3 predictions with confidence scores
## Model Architecture
GarDoc uses a transfer learning approach with the following architecture:

- Base model: ResNet50 pre-trained on ImageNet
- Custom top layers:
  - Global Average Pooling
  - Dense layer (1024 units, ReLU activation)
  - Dropout (0.5)
  - Dense layer (512 units, ReLU activation)
  - Dropout (0.3)
  - Output layer (softmax activation)
The model is trained in two phases:

1. Training only the custom top layers
2. Fine-tuning the last 30 layers of ResNet50 along with the top layers
## Dataset Requirements
For optimal results:

- Use high-quality, well-lit images
- Ensure images clearly show the plant and any disease symptoms
- Include multiple angles and perspectives for each class
- Aim for at least 100 images per class for good performance
- Images will be resized to 224×224 pixels during processing
## Troubleshooting
### Common Issues
1. ImportError: No module named 'tensorflow'
   
   - Ensure you've activated the virtual environment
   - Reinstall dependencies: pip install -r requirements.txt
2. GPU-related errors
   
   - The model can run on CPU, but for faster training, a compatible GPU is recommended
   - For GPU support, ensure you have the correct CUDA and cuDNN versions installed
3. Memory errors during training
   
   - Reduce the batch size in train_model.py
   - Use a smaller subset of your dataset
4. Model prediction errors
   
   - Ensure the class mapping file matches the model
   - Check that the image format is supported (JPG, PNG)
For additional help, please open an issue on the repository.

```plaintext

This README provides comprehensive instructions for setting up the environment, training the model, and making predictions with your plant disease classification system. It includes details about the project structure, model architecture, and troubleshooting tips to help users get started quickly.
 ```
```