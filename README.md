# Real-time Hand Sign Detection

## Description
This project is designed to detect hand signs in real-time using your webcam. It uses MediaPipe for hand landmark detection, and trains a Random Forest Classifier for the hand sign classification.

## Installation
1. Clone this repository: 
git clone https://gitlab.dev.info.uvt.ro/didactic/2023/licenta/IE/licenta-alin-preda01-2023.git
2. Install the required packages: 
pip install -r requirements.txt

## Usage
I wanted to provide the model so that inference_classifier would be enough to run the project but GitLab didn't allow me to upload it.
The project involves several steps:


1. **Image Collection**: Run `collect_imgs.py` to collect images for each class of hand signs. This script uses your webcam to capture images and save them in a specified directory.

2. **Dataset Creation**: Run `create_dataset.py` to process the collected images and create a dataset. This script uses MediaPipe to extract hand landmarks from the images and prepares a dataset that can be used for model training.

3. **Model Training**: Run `train_classifier.py` to train a Random Forest Classifier on the created dataset. This script also evaluates the model's accuracy on a test set and saves the trained model for later use.

4. **Real-time Hand Sign Detection**: Run `inference_classifier.py` to use the trained model for real-time hand sign detection. This script uses your webcam to capture video, processes each frame to detect hand signs, and uses the trained model to predict the class of the hand sign. The predicted character is displayed inside a chat icon.

## License
This project is licensed under the terms of the MIT License.
