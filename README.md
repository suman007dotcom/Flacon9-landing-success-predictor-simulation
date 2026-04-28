# Falcon 9 Landing Predictor

This project is a machine learning web app made using Streamlit that predicts whether a SpaceX Falcon 9 booster will land successfully after launch. The idea of this project is based on real Falcon 9 launch data and uses classification models to make predictions.
The app is interactive and user friendly. Users can enter mission details like payload mass, orbit type, launch site, booster version, number of previous flights and more. After that, the model predicts if the booster will land safely or fail.

# About the Project

Falcon 9 is famous because SpaceX reuses rocket boosters after launch. Safe landing helps reduce launch costs. In this project, machine learning is used to study past launch patterns and predict future landing success.
The project compares two models:
Logistic Regression, Random Forest Classifier

The better performing model is selected automatically for final prediction.

# Features

Modern dark themed Streamlit interface

Interactive launch simulation

Real time landing prediction

Confidence score for predictions

Comparison of two machine learning models

Charts and visualizations using Plotly

Feature importance analysis

Historical success trends over years

# Technologies Used

Python

Streamlit

Pandas

NumPy

Scikit Learn

Plotly

# Dataset

The dataset contains Falcon 9 launch records with features such as

Payload Mass

Launch Site

Orbit

Booster Version

Grid Fins

Landing Legs

Flight Number

Reuse Count

Outcome

# Machine Learning Workflow

Data cleaning and preprocessing

Handling missing values

Encoding categorical variables

Train test split

Feature scaling using StandardScaler

Training Logistic Regression and Random Forest

Accuracy comparison

Best model selection

Prediction on user input

# Pages in App

Home Page

Shows project overview, model accuracy, real launch examples, prediction performance and yearly success trends.

Feature Guide

Explains each input feature in simple language and shows importance of features used by the model.

Simulate a Launch

Users can create their own mission scenario and get landing prediction instantly.

# How to Run

Install required libraries first

pip install streamlit pandas numpy scikit-learn plotly

Run the app

streamlit run app2.py

# Learning Outcome

Through this project, I learned how to build complete machine learning projects, used Claude help to learn streamlit needed for the project.
Handle preprocessing and feature engineering.
Compare multiple classification models.
Create interactive dashboards.

Use data science for real world aerospace problems

# Conclusion

This project combines machine learning with space technology in an engaging way. It shows how historical data can be used to predict rocket landing success and demonstrates practical use of data science concepts.
