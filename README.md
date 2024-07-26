# Bike-Sharing-Demand
Overview
This project aims to build a multiple linear regression model to predict the demand for shared bikes. The model helps in understanding the factors affecting bike rental demand and assists the company in strategizing to meet customer needs and increase revenue.

Business Problem
BoomBikes, a US-based bike-sharing provider, has experienced a decline in revenues due to the COVID-19 pandemic. The company seeks to understand the factors influencing bike rental demand to prepare for the post-pandemic market. This model will be used to predict bike rental demand based on various factors and help the company in decision-making.

Dataset
The dataset contains daily records of bike rentals along with various features such as weather conditions, season, holiday, and more. The target variable is cnt, which represents the total number of bike rentals.

Project Structure
data/day.csv: The dataset used for building the model.
notebooks/Bike_Sharing_Demand_Prediction.ipynb: Jupyter notebook containing the code for data exploration, preparation, model building, and evaluation.
README.md: This file, providing an overview of the project.
Steps
1. Data Loading and Understanding
Load the dataset and display the first few rows, information, and summary statistics.
2. Data Visualization
Visualize the distribution of the target variable (cnt).
Explore the relationship between numeric features (temp, atemp, hum, windspeed) and the target variable.
Explore the relationship between categorical features (season, yr, mnth, holiday, weekday, workingday, weathersit) and the target variable.
3. Data Preparation
Convert numeric codes to categorical strings for season and weathersit.
Drop unnecessary columns (instant, dteday, casual, registered).
Encode categorical variables.
4. Train-Test Split
Split the data into training and testing sets.
5. Model Building
Build a multiple linear regression model using the training data.
6. Residual Analysis
Perform residual analysis on the training data to check for any patterns.
7. Model Evaluation
Make predictions on the test data.
Evaluate the model using R-squared score.
Display the model's coefficients and intercept.
