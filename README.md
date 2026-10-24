
# Handwritten Digit Recognition Using CNN

## Overview
A deep learning application that recognizes handwritten
digits from 0 to 9 using a Convolutional Neural Network
implemented with PyTorch.

## Features
- CNN training on the MNIST dataset
- Model evaluation and test accuracy
- Precision, recall and F1-score
- Confusion matrix visualization
- Analysis of incorrect predictions
- Interactive digit drawing and prediction
- Probability visualization for all ten digits

## Tech Stack
- Python
- PyTorch
- Torchvision
- Matplotlib
- Scikit-learn
- Streamlit

## Project Structure
- model.py: CNN architecture
- train.py: Model training
- evaluate.py: Evaluation and visualization
- app.py: Interactive web application
- models/: Saved model weights
- results/: Evaluation plots and training history

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Train the model:
   python train.py

3. Evaluate the model:
   python evaluate.py

4. Launch the application:
   streamlit run app.py

## Dataset
MNIST handwritten digit dataset, containing
60,000 training images and 10,000 test images.

## Future Improvements
- Compare multiple CNN architectures
- Improve preprocessing for user-drawn digits
- Test on handwritten images outside MNIST
- Experiment with model size and training parameters
# handwritten-digit-recognition
