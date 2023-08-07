# Computer Vision Project: Landmark Detection and Model Building
## Project Overview
This computer vision project utilizes OpenCV and MediaPipe libraries to detect facial landmarks in images and videos. The detected landmarks are then used to train a machine learning model for further analysis.

## File Structure
The project consists of the following files:

[app_dataCollection](./app_dataCollection.ipynb): This Jupyter Notebook is used for collecting facial landmarks data. It utilizes OpenCV and MediaPipe to detect facial landmarks in images or videos and saves the landmarks data for further processing.

[Data_proc_Model_building](./Data_proc_Model_building.ipynb): This Jupyter Notebook is responsible for processing the collected landmarks data and building the machine learning model. It involves data preprocessing, feature engineering, model training, and testing.

model1/: This directory contains the saved machine learning model. The trained model is stored here for later use and inference.

## Dependencies
The project relies on the following libraries:

OpenCV: A computer vision library for image and video processing.
MediaPipe: A library that offers a variety of pre-trained models for tasks like face detection, pose estimation, and hand tracking.
Other common Python libraries used for data manipulation and model building, such as NumPy, pandas.

## Usage
Data Collection: Run app_dataCollection.ipynb to collect facial landmarks data. The notebook provides functions to load images or videos, detect facial landmarks using MediaPipe, and save the detected landmarks as a dataset for future use.

Data Processing and Model Building: After collecting the landmarks data, run Data_proc_Model_building.ipynb. This notebook handles data preprocessing, feature engineering, model training, and evaluation. The trained model is then saved in the model1/ directory.

Inference: Once the model is trained and saved, you can use it to make predictions on new data. You may load the model and use it together with OpenCV to see a live prediction

## Note
Make sure to have all the required libraries installed before running the notebooks.
Adjust the parameters and configurations according to your specific use case.
The project can be further extended to other tasks like emotion recognition or facial expression analysis.
## Contact
If you have any questions or suggestions, feel free to reach out to the project maintainers:

[Ajibola Matthew](www.linkedin.com/in/jibbycodes)
