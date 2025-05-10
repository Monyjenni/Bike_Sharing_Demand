# Bike Sharing Demand Prediction

<img width="1375" alt="Screenshot 2025-04-08 at 10 41 20 at night" src="https://github.com/user-attachments/assets/a215cc73-9464-4d12-b6f2-d160a409f062" />


<img width="873" alt="Screenshot 2025-04-08 at 10 41 49 at night" src="https://github.com/user-attachments/assets/06bc3844-e76b-446d-ae3e-f766c9e982bc" />


<img width="495" alt="Screenshot 2025-04-08 at 10 42 04 at night" src="https://github.com/user-attachments/assets/5330dc29-e7d4-4b6e-98af-291deabb4f3d" />

## Project Overview

This project aims to predict the number of bikes rented on a given day for the Capital Bikeshare program in Washington, D.C. using machine learning techniques. The model incorporates various features including weather conditions, time-based features, and seasonal patterns to make accurate predictions.

### Competition Description

In this competition, participants are asked to combine historical usage patterns with weather data to forecast bike rental demand. The evaluation metric is Root Mean Squared Logarithmic Error (RMSLE), which penalizes underestimation more than overestimation.

## Dataset Description

The training dataset contains historical bike rental data with the following features:

- `datetime`: Date and time of the record
- `season`: Season (1=spring, 2=summer, 3=fall, 4=winter)
- `holiday`: Whether the day is a holiday (0=no, 1=yes)
- `workingday`: Whether the day is a working day (0=no, 1=yes)
- `weather`: Weather condition (1=clear, 2=mist, 3=light rain/snow, 4=heavy rain/snow)
- `temp`: Temperature in Celsius (normalized)
- `atemp`: "Feels like" temperature in Celsius (normalized)
- `humidity`: Relative humidity (normalized)
- `windspeed`: Wind speed (normalized)
- `casual`: Count of casual users
- `registered`: Count of registered users
- `count`: Total count of bike rentals (target variable)

The test dataset contains the same features except for the target variable (`count`), which we need to predict.

## Setup Instructions

### Environment Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Notebook

1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

2. Open `bike-sharing-ml.ipynb` and run the cells in sequence.

## Model Features

### Data Preprocessing
- Datetime feature extraction (year, month, day, hour, etc.)
- Cyclic encoding of time features
- Rush hour identification

### Feature Engineering
- Weather interaction features
- Comfort metrics
- Extreme condition flags
- Seasonal indicators

### Modeling Approach
- Gradient Boosting Regressor with optimized hyperparameters
- Log transformation of the target variable
- Comprehensive evaluation using RMSLE, RMSE, and R² metrics

## Results

The Gradient Boosting model achieves strong performance by effectively capturing:

1. **Temporal patterns**: Hour of day, day of week, and seasonal trends
2. **Weather impact**: Temperature, humidity, and weather condition effects
3. **Feature interactions**: Combined effects of multiple variables

The model's predictions are saved to `bike_sharing_predictions.csv`, which is ready for submission to the competition.

## Project Structure

- `Data/`: Contains the training and test datasets
- `bike-sharing-ml.ipynb`: Jupyter notebook with the complete analysis and model
- `requirements.txt`: List of required Python packages
- `model_explanation.md`: Detailed explanation of the model approach
- `bike_sharing_predictions.csv`: Final predictions for submission

## Detailed Documentation

For a comprehensive explanation of the model, including data preprocessing, feature engineering, model training, and evaluation, please refer to the `model_explanation.md` file.
