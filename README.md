# DeepFake Detection

A deep learning project for detecting deepfake images using Convolutional Neural Networks.

## Overview

This project builds a CNN-based classifier (ResNet50) to distinguish between real and deepfake images.  
It leverages transfer learning and image preprocessing techniques to achieve high accuracy on a large-scale dataset (~140k images).

## Tech Stack

- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy
- Flask (for optional web interface)

## Project Structure

```
deep_fake_project/
├── Dataset/                 # Image dataset
├── app.py                  # Flask app for prediction
├── model_build.py          # Model architecture
├── model_train.py          # Training script
├── evaluate_model.ipynb    # Evaluation and visualization
├── utils.py                # Helper functions
├── model.h5                # Trained model (not in repo)
├── .gitignore
```
