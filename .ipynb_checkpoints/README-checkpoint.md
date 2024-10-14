# AAI540_MLOps_Final_Team4
University of San Diego - AAI540 MLOps - Final Project - Team 4


# Skin Cancer Prediction using MlOps/Computer Vision

## Project Background

Asserting the difference between benign and malignant tumors for skin lesions can be difficult to differentiate even for many doctors at a glance. Creating an application that can detect malignant vs benign skin tumors can be integral in helping possible patients detect cancer at a much earlier stage and give them a higher possibility for successful treatment. 

The model will analyze the skin lesion and determine whether it is benign of malignant, a binary classification using supervised learning techniques. 

## Technical Background

We will be using the Skin Cancer Dataset: https://www.kaggle.com/datasets/shashanks1202/skin-cancer-dataset 

The dataset is organized into two folders:

* Malignant: This folder contains images of skin lesions that have been diagnosed as malignant, indicating the presence of skin cancer. These images can be used for training models to detect and differentiate malignant skin conditions.

* Benign: This folder includes images of benign skin lesions, which are non-cancerous and pose no immediate threat. These images are essential for training models to accurately distinguish between harmful and harmless skin conditions (Kaggle Shashank S). 

Data preparation is a crucial step in this process as we need to capture very minute details that can be the tell between the malignant and benign tumors. So we will try to regularize the pictures with resizing them and possible converting the color palette. We will split up the data into training and test as well.

We will most likely use a ResNet model, CNN, to do the predictions. We can test other models as well like simple linear regression.

We can explore the datapoints by reducing the picutre to x/y coordinates using dimension reduction and plotting these values to help in the prediction. 

We will evaluate the model using traditional means like accuracy, f1, mean, precision, recall.

## Goals vs Non-Goals

Goals:

* Create an accurate model that is not overfitted to the data but can give strong confidence
* Allow the model to have higher accuracy scores, as false negatives are a worse situation for the model due to the sensitivity of the issue
* Use mulitple types of models and save the models in a model repo to see which model can perform the best

Non-Goals:

* Optimize model performance to be efficient as possible - i do not see this as integral as accuracy of model is most important here
* Allow the model to take in diff inputs, pictures only for now
