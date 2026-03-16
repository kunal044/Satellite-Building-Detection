# Satellite Building Detection using Deep Learning

This project focuses on automatically detecting building infrastructure from high-resolution satellite images using deep learning techniques. The goal is to identify building areas in satellite imagery and generate building masks that highlight detected structures.

This project is developed as part of the **Design Thinking and Innovation course at Bennett University**.

---

## Project Overview

Satellite images contain a large amount of geographic information. Manually identifying buildings from these images is time-consuming and prone to errors. This project aims to automate the process using **semantic segmentation models**.

Deep learning models analyze each pixel in the satellite image and classify it as:

- Building
- Non-Building

The system processes satellite images and generates predicted building masks that highlight the detected building regions.

---

## Technologies Used

- MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- Satellite Image Dataset
- Semantic Segmentation

---

## Deep Learning Models

The project implements and compares two popular segmentation architectures:

### U-Net
U-Net is an encoder-decoder based convolutional neural network commonly used for image segmentation tasks.

### ResUNet
ResUNet improves the U-Net architecture by introducing residual connections, which help improve gradient flow and model performance.

---

## Project Workflow

1. **Satellite Image Collection**  
   Collect high-resolution satellite images and corresponding ground truth masks.

2. **Data Preprocessing**  
   - Image normalization  
   - Data augmentation (rotation, flipping)

3. **Dataset Splitting**  
   - 80% Training  
   - 10% Validation  
   - 10% Testing

4. **Model Training**  
   Train U-Net and ResUNet models on the dataset.

5. **Prediction**  
   The trained model predicts building areas in satellite images.

6. **Evaluation**  
   Evaluate model performance using metrics such as:
   - Precision
   - Recall
   - F1-Score
   - Intersection over Union (IoU)

---

## Applications

This system can be useful in many real-world applications:

- Urban planning
- Smart city development
- Infrastructure monitoring
- Disaster management
- Geographic information systems (GIS)

---

## Repository Structure
