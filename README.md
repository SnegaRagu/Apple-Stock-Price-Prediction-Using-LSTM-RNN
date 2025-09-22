# Introduction

Project focuses on predicting the future stock price of Apple (AAPL) using deep learning techniques. Analysis is performed on Apple’s adjusted closing price, and we aim to build two deep learning models — a SimpleRNN and an LSTM — to forecast stock price movements. The models are trained to predict the adjusted closing price over three forecast horizons: 1 day ahead, 5 days ahead, and 10 days ahead. Finally, we compare the performance of the models to evaluate which approach is better suited for short- and medium-term stock price forecasting.

# Project Overview

## 1. Problem Understanding

> The goal is to predict Apple stock prices using historical data.
> We will use a deep learning-based approach (SimpleRNN and LSTM) to model stock price trends.
> The dataset consists of features: Date, Open, High, Low, close, Adj adj close, and Volume.

## 2. Data Preprocessing

### 2.1. Load Dataset

> Read the Apple stock price dataset (AAPL.csv) using pandas.
> Explore the dataset to understand key features.

### 2.2. Feature Selection

> Use the adj close price as the target variable for training the model.
> Convert the Date column to a datetime format if necessary and set it as the index.

### 2.3. Scaling the Data

> Apply MinMaxScaler (normalization) to scale stock prices between 0 and 1 for better model convergence.

### 2.4. Creating Time-Series Sequences

> Prepare the data for LSTM by creating input-output sequences.
> Use a window of past n days to predict the next stock price.

## 3. Model Development

### 3.1. Define SimpleRNN & LSTM Architecture

> Use Sequential from tensorflow.keras to build a SimpleRNN/LSTM model.
> Layers used:
  > SimpleRNN/LSTM layer for learning sequential dependencies.
  > Dropout layer to prevent overfitting.
  > Dense layer to output the predicted stock price.

### 3.2. Compile the Model

> Use mean_squared_error (MSE) as the loss function.
> Optimize using Adam optimizer.

### 3.3. Model Training

> Train the model with early stopping to avoid overfitting.
> Use ModelCheckpoint to save the best model.

## 4. Model Evaluation & Prediction

> Use the trained SimpleRNN/LSTM model to predict stock prices on the test set.
> Compare actual vs predicted stock prices using visualization (matplotlib).
> Calculate Mean Squared Error (MSE) to evaluate model performance.

## 5. Business Cases

### 5.1 Feature Engineering & Alternative Data

> Enhance the model by adding news sentiment analysis, social media trends, volume trends or macroeconomic indicators.

### 5.2 Comparing Time-Series Models

> Extend the project by comparing LSTMs with GRU.

### 5.3 Macroeconomic Analysis

> Compare Apple’s stock trends with economic indicators like interest rates, inflation, and industry trends.

# Results

## Model Performance Results (MSE)

| Model      | Dataset                          | Horizon | MSE   |
|:----------:|:--------------------------------:|:--------:|:-------:|
| SimpleRNN | OHLCV                            | 1-Day  | 48.6 |
| SimpleRNN | OHLCV                            | 5-Day  | 132.4 |
| SimpleRNN | OHLCV                            | 10-Day | 215.9 |
| LSTM      | OHLCV                            | 1-Day  | 49.1 |
| LSTM      | OHLCV                            | 5-Day  | 107.6 |
| LSTM      | OHLCV                            | 10-Day | 182.6 |
| GRU       | OHLCV                            | 1-Day  | **13.1** |
| GRU       | OHLCV                            | 5-Day  | **45.3** |
| GRU       | OHLCV                            | 10-Day | **114.3** |
| SimpleRNN | OHLCV + Extra Features            | 1-Day  | 64.7 |
| SimpleRNN | OHLCV + Extra Features            | 5-Day  | 132.3 |
| SimpleRNN | OHLCV + Extra Features            | 10-Day | 159.6 |
| LSTM      | OHLCV + Extra Features            | 1-Day  | 403.7 |
| LSTM      | OHLCV + Extra Features            | 5-Day  | 464.1 |
| LSTM      | OHLCV + Extra Features            | 10-Day | 576.3 |
| GRU       | OHLCV + Extra Features            | 1-Day  | 165.5 |
| GRU       | OHLCV + Extra Features            | 5-Day  | 150.9 |
| GRU       | OHLCV + Extra Features            | 10-Day | 328.2 |


> **GRU** is the most robust model for predicting Apple’s stock price using OHLCV data, especially for short-term forecasts (1-day and 5-day horizons). However, integrating additional features beyond OHLCV requires careful preprocessing and feature selection to avoid model overfitting and performance degradation, particularly for complex models like LSTMs.

