# Cycle Theft Detection Model

## Overview

This project showcases an efficient Cycle Theft Detection Model that leverages the power of Autoencoders, LSTMs, and advanced Anomaly Detection techniques. Our model performs effectively with good processing speed and maintains high reliability, even under challenging conditions such as low light and suboptimal image quality. This versatility allows it to perform effectively across a wide range of datasets.

## Methodology

### 1. Data Collection and Preprocessing

**Dataset**: The dataset consists of video footage containing various activities, including normal and suspicious events related to cycle theft.

**Preprocessing**:

- Video frames were extracted and resized for compatibility with the model.
- Frame sequences were generated for training the LSTM component.

### 2. Model Architecture

**Autoencoder**:

- A convolutional autoencoder was used for feature extraction. This architecture aids in compressing the input frames into a lower-dimensional representation.

**LSTM**:

- The LSTM component processed sequences of features extracted by the autoencoder, capturing temporal information to identify patterns in suspicious activities.

**Training**:

- The model was trained using a combination of reconstruction loss from the autoencoder and anomaly scores generated by the LSTM.

### 3. Anomaly Detection

**Anomaly Scores**:

- Anomaly scores were computed based on the difference between the input frames and their reconstructions. Higher scores indicate greater deviation from normal activities.

**Detection Algorithm**:

- An algorithm was developed to identify heightened periods by considering moving windows of anomaly scores. It dynamically adjusted the threshold to adapt to changing trends.

### 4. Model Evaluation

**Validation**:

- The model was validated on a separate dataset to assess its performance in detecting cycle theft events.

## Usage

**Training**:

- To train the model, refer the `train.ipynb` file. Make sure to specify the appropriate paths and hyperparameters.

**Inference**:

- The `inference.py` script is used for applying the trained model to new video footage. Specify the model path, video path, and output paths.

**Anomaly Analysis**:

- The `detect_anomalies` method in `inference.py` identifies heightened periods based on anomaly scores. We also have a separate `detect_anomalies.py` for analysing the trends. The results containing the suspicious periods are stored in the `anomalies.json` file under the results folder.

## GitHub Repository Structure

- `data/`: Directory containing the dataset (video footages).
- `models/`: Directory to store trained model files.
- `results/`: Directory for storing output files (JSON, plots, etc.).
- `train.ipynb`: Script for training the model.
- `inference.py`: Script for applying the trained model to new video footage.


## How to Run

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Follow the usage instructions provided in the README.

## Conclusion

The Cycle Theft Detection Model combines the power of Autoencoders, LSTMs, and Anomaly Detection to identify suspicious events in video footage. By leveraging deep learning techniques, this model offers a reliable solution for enhancing security and preventing cycle theft incidents even in low light and poor image quality conditions.
