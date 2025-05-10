# Bike Sharing Demand Prediction Model - Detailed Explanation

This document provides a comprehensive explanation of the bike sharing demand prediction model implementation, including data preprocessing, exploratory data analysis, feature engineering, modeling, and evaluation.

## 1. Project Overview

**Objective**: Predict the number of bikes rented on a given day for the Capital Bikeshare program in Washington, D.C.

In this competition, participants are asked to combine historical usage patterns with weather data to forecast bike rental demand. The model uses various features such as weather conditions, time-based features, and seasonal patterns to make accurate predictions.

## 2. Data Loading and Initial Exploration

```python
# Load the datasets
train_data = pd.read_csv('../Data/train.csv')
test_data = pd.read_csv('../Data/test.csv')
```

The model begins by loading two datasets:
- **Training dataset**: Contains historical bike rental data with features and the target variable (count of bikes rented)
- **Test dataset**: Contains similar features but without the target variable, used for making predictions

The training data includes these key columns:
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

## 3. Data Preprocessing

### 3.1 Missing Value Check

```python
train_data.isnull().sum()
test_data.isnull().sum()
```

The code checks for missing values in both datasets. Fortunately, no missing values were found, so no imputation was needed.

### 3.2 Datetime Feature Extraction

```python
def extract_datetime_features(df):
    # Basic datetime features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # Derived datetime features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['datetime'].dt.quarter
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['weekofyear'] = df['datetime'].dt.isocalendar().week
    
    # Time period categories
    time_periods = {
        'night': (0, 6),
        'morning': (6, 12),
        'afternoon': (12, 18),
        'evening': (18, 24)
    }
    
    # Create time of day feature
    conditions = [
        (df['hour'] >= time_periods['night'][0]) & (df['hour'] < time_periods['night'][1]),
        (df['hour'] >= time_periods['morning'][0]) & (df['hour'] < time_periods['morning'][1]),
        (df['hour'] >= time_periods['afternoon'][0]) & (df['hour'] < time_periods['afternoon'][1]),
        (df['hour'] >= time_periods['evening'][0]) & (df['hour'] < time_periods['evening'][1])
    ]
    choices = ['night', 'morning', 'afternoon', 'evening']
    df['time_of_day'] = np.select(conditions, choices, default='unknown')
    
    # Create cyclic features for time variables
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # Rush hour flags
    morning_rush = (df['hour'] >= 7) & (df['hour'] <= 9)
    evening_rush = (df['hour'] >= 16) & (df['hour'] <= 19)
    df['is_rush_hour'] = (morning_rush | evening_rush).astype(int)

    # Weekend rush hour (different pattern on weekends)
    weekend_noon_rush = (df['hour'] >= 11) & (df['hour'] <= 14)
    weekend_evening_rush = (df['hour'] >= 16) & (df['hour'] <= 20)
    df['is_weekend_rush'] = ((df['is_weekend'] == 1) & (weekend_noon_rush | weekend_evening_rush)).astype(int)
    
    return df
```

This function extracts valuable features from the datetime column:

- **Basic time components**: Year, month, day, hour, day of week
- **Derived time features**: Weekend flag, quarter, day of year, week of year
- **Time period categorization**: Night (0-6h), morning (6-12h), afternoon (12-18h), evening (18-24h)
- **Cyclic features**: Sine and cosine transformations of time variables to capture their cyclical nature
- **Rush hour flags**: Morning rush (7-9h), evening rush (16-19h), and weekend-specific rush hours

Cyclic features are particularly important for time data because they preserve the circular relationship (e.g., hour 23 is close to hour 0, December is close to January). This helps the model understand the continuity in time variables.

## 4. Feature Engineering

```python
def feature_engineering(df):
    # Interaction features
    df['temp_humidity'] = df['temp'] * df['humidity']
    df['temp_windspeed'] = df['temp'] * df['windspeed']
    df['humidity_windspeed'] = df['humidity'] * df['windspeed']
    df['atemp_humidity'] = df['atemp'] * df['humidity']
    df['atemp_windspeed'] = df['atemp'] * df['windspeed']
    
    # Temperature difference (feels like vs actual)
    df['temp_diff'] = df['atemp'] - df['temp']
    
    # Weather severity
    weather_severity_map = {1: 1, 2: 2, 3: 3, 4: 4}
    df['weather_severity'] = df['weather'].map(weather_severity_map)
    
    # Extreme weather conditions
    df['is_extreme_temp'] = ((df['temp'] < 0.2) | (df['temp'] > 0.8)).astype(int)
    df['is_extreme_humidity'] = ((df['humidity'] < 0.2) | (df['humidity'] > 0.8)).astype(int)
    df['is_extreme_windspeed'] = (df['windspeed'] > 0.5).astype(int)
    df['is_bad_weather'] = (df['weather'] >= 3).astype(int)
    
    # Ideal biking condition (moderate temp, low humidity, low windspeed, good weather)
    df['ideal_biking_condition'] = (((df['temp'] >= 0.4) & (df['temp'] <= 0.7)) & 
                                  (df['humidity'] < 0.6) & 
                                  (df['windspeed'] < 0.3) & 
                                  (df['weather'] <= 2)).astype(int)
    
    # Comfort metrics
    df['comfort_index'] = df['temp'] - 0.1 * df['humidity'] - 0.1 * df['windspeed']
    
    # Season-based features
    df['is_summer'] = (df['season'] == 2).astype(int)
    df['is_fall'] = (df['season'] == 3).astype(int)
    df['is_winter'] = (df['season'] == 4).astype(int)
    df['is_spring'] = (df['season'] == 1).astype(int)
    
    # Holiday and working day interaction
    df['free_day'] = ((df['holiday'] == 1) | (df['is_weekend'] == 1)).astype(int)
    
    return df
```

The feature engineering function creates several types of derived features:

1. **Interaction features**: Combinations of existing features that might have multiplicative effects (e.g., temperature × humidity)
2. **Temperature difference**: Captures the difference between actual and "feels like" temperature
3. **Weather severity**: Numerical representation of weather conditions
4. **Extreme condition flags**: Binary indicators for extreme weather conditions
5. **Ideal biking conditions**: Flag for ideal biking conditions based on multiple criteria
6. **Comfort metrics**: Custom formula to estimate biking comfort based on weather conditions
7. **Season-based features**: Binary indicators for each season
8. **Free day indicator**: Combines holidays and weekends into a single feature

These engineered features help the model understand complex patterns and interactions in the data that affect bike rental demand.

## 5. Exploratory Data Analysis (EDA)

The EDA section creates several visualizations to understand the data patterns:

### 5.1 Distribution of Bike Rentals

```python
plt.figure(figsize=(12, 6))
sns.histplot(train_data['count'], kde=True)
plt.title('Distribution of Bike Rentals Count')
plt.xlabel('Count')
plt.ylabel('Frequency')
```

**Analysis**: The histogram shows the distribution of bike rental counts. The distribution is right-skewed, with most rentals falling in the lower range (0-200 bikes), but with a long tail extending to higher values. This skewness suggests that log transformation of the target variable might be beneficial for modeling.

### 5.2 Hourly Rental Patterns

```python
plt.figure(figsize=(14, 7))
sns.boxplot(x='hour', y='count', data=train_data)
plt.title('Bike Rentals by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Count')
```

**Analysis**: This boxplot reveals strong hourly patterns in bike rentals:
- Two distinct peaks are visible: morning rush hour (7-9 AM) and evening rush hour (5-6 PM)
- The highest median rentals occur during the evening rush hour
- Nighttime hours (0-5 AM) show very low rental activity
- There's significant variability (large boxes) during peak hours, indicating that other factors (like weather or day of week) also influence rentals during these times

These patterns strongly align with commuting behavior, suggesting that many bike rentals are for commuting to and from work.

### 5.3 Seasonal Rental Patterns

```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='season', y='count', data=train_data)
plt.title('Bike Rentals by Season')
plt.xlabel('Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)')
plt.ylabel('Count')
```

**Analysis**: The seasonal boxplot shows:
- Summer (2) and Fall (3) have the highest median bike rentals
- Winter (4) has the lowest median rentals
- Spring (1) shows moderate rental activity
- Fall has the highest variability, suggesting other factors (like weather events) have significant influence during this season

This pattern is expected as weather conditions in summer and fall are generally more favorable for biking compared to winter and early spring.

### 5.4 Weather Impact on Rentals

```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='weather', y='count', data=train_data)
plt.title('Bike Rentals by Weather Condition')
plt.xlabel('Weather (1=Clear, 2=Mist, 3=Light Rain/Snow, 4=Heavy Rain/Snow)')
plt.ylabel('Count')
```

**Analysis**: This visualization clearly shows the impact of weather on bike rentals:
- Clear weather (1) has the highest median rentals
- Misty conditions (2) show slightly lower rentals
- Light rain/snow (3) shows a significant drop in rentals
- Heavy rain/snow (4) has the lowest rentals, with very few outliers

This confirms our intuition that adverse weather conditions significantly reduce bike rental demand.

### 5.5 Correlation Heatmap

```python
plt.figure(figsize=(16, 12))
key_features = ['count', 'temp', 'atemp', 'humidity', 'windspeed', 'hour', 'dayofweek', 
                'month', 'season', 'weather', 'holiday', 'workingday']
correlation = train_data[key_features].corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Correlation Heatmap of Key Features')
```

**Analysis**: The correlation heatmap reveals important relationships:
- Temperature (`temp` and `atemp`) has the strongest positive correlation with bike rentals
- Humidity has a moderate negative correlation with rentals
- Hour of day shows a notable correlation, confirming the importance of time patterns
- Weather condition has a negative correlation (worse weather = fewer rentals)
- There's a strong correlation between `temp` and `atemp`, suggesting we might not need both
- Season and month are correlated, as expected

These correlations guide our understanding of which features are most important for predicting bike rentals.

## 6. Data Preparation for Modeling

```python
# Define features to use
features = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend', 'quarter',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos',
    'is_rush_hour', 'is_weekend_rush', 'temp_humidity', 'temp_windspeed', 'humidity_windspeed',
    'atemp_humidity', 'atemp_windspeed', 'temp_diff', 'weather_severity',
    'is_extreme_temp', 'is_extreme_humidity', 'is_extreme_windspeed', 'is_bad_weather',
    'ideal_biking_condition', 'comfort_index', 'is_summer', 'is_fall', 'is_winter', 'is_spring', 'free_day'
]

# Prepare training data
X = train_data[features]
y = train_data['count']

# Log transform the target variable
y_log = np.log1p(y)

# Split the data into training and validation sets
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
```

The data preparation steps include:

1. **Feature selection**: Choosing the most relevant features, including original features and engineered ones
2. **Target transformation**: Applying log transformation (log1p) to the target variable to handle skewness
3. **Train-validation split**: Splitting the data into training (80%) and validation (20%) sets

The log transformation is particularly important because:
- It helps normalize the right-skewed distribution of bike rentals
- It aligns with the competition's evaluation metric (RMSLE), which operates on log-transformed values
- It helps the model better handle the wide range of rental counts

## 7. Model Training

```python
# Train the enhanced Gradient Boosting model
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train_log)
```

For the final model, we chose the Gradient Boosting Regressor because:

1. **Handling non-linear relationships**: Gradient boosting excels at capturing complex, non-linear patterns in the data
2. **Feature importance**: It provides useful insights into which features are most influential
3. **Robustness**: It's less prone to overfitting compared to some other tree-based methods
4. **Performance**: It consistently outperformed other models in our experiments

The hyperparameters were carefully tuned:
- `n_estimators=300`: Number of boosting stages (trees)
- `learning_rate=0.05`: Shrinks the contribution of each tree (helps prevent overfitting)
- `max_depth=5`: Maximum depth of individual trees (controls complexity)
- `min_samples_split=5`: Minimum samples required to split a node
- `min_samples_leaf=2`: Minimum samples required in a leaf node
- `subsample=0.8`: Fraction of samples used for fitting individual trees (helps prevent overfitting)

## 8. Model Evaluation

```python
# Make predictions on validation set
gb_pred = np.expm1(gb.predict(X_val))

# Evaluate the model
rmsle_score = rmsle(y_val, gb_pred)
rmse_score = np.sqrt(mean_squared_error(y_val, gb_pred))
r2_score_val = r2_score(y_val, gb_pred)

print(f"Enhanced Model RMSLE: {rmsle_score:.4f}")
print(f"Enhanced Model RMSE: {rmse_score:.4f}")
print(f"Enhanced Model R²: {r2_score_val:.4f}")
```

The model is evaluated using multiple metrics:

1. **RMSLE (Root Mean Squared Logarithmic Error)**: The primary metric for the competition, which penalizes underestimation more than overestimation and handles the wide range of rental counts well
2. **RMSE (Root Mean Squared Error)**: A standard regression metric that gives an idea of the average prediction error
3. **R² (Coefficient of Determination)**: Indicates the proportion of variance in the target variable that the model explains

The model achieves strong performance on the validation set, with a low RMSLE and high R² value, indicating good predictive power.

### 8.1 Actual vs. Predicted Values

```python
plt.figure(figsize=(12, 8))
plt.scatter(y_val, gb_pred, alpha=0.5)
plt.plot([0, max(y_val)], [0, max(y_val)], 'r--')
plt.title('Actual vs Predicted Values - Enhanced Gradient Boosting')
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

**Analysis**: This scatter plot compares actual rental counts with the model's predictions on the validation set:
- Points close to the red diagonal line represent accurate predictions
- The model performs well across most of the range, with points clustered around the diagonal
- There's some scatter at higher rental counts, indicating slightly less accuracy for very high demand days
- No systematic bias is visible (points are evenly distributed above and below the diagonal)

This visualization confirms that the model makes reasonable predictions across the range of rental counts.

### 8.2 Feature Importance

```python
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance - Enhanced Gradient Boosting')
```

**Analysis**: The feature importance plot reveals which features contribute most to the model's predictions:
- Temporal features (hour, hour_sin, hour_cos) are among the most important, confirming the strong daily patterns
- Weather-related features (temp, humidity) also rank highly
- Several engineered features appear in the top 20, validating our feature engineering approach
- The importance of both original and cyclic time features suggests that the model effectively captures temporal patterns

This analysis helps us understand the key drivers of bike rental demand and confirms that our feature engineering was effective.

## 9. Making Predictions on Test Data

```python
# Make predictions on test data
X_test = test_data[features]
final_predictions = np.expm1(gb.predict(X_test))

# Create submission file
submission = pd.DataFrame({
    'datetime': test_data['datetime'],
    'count': final_predictions
})

# Ensure count values are non-negative
submission['count'] = submission['count'].clip(lower=0)
# Round predictions to integers as the competition requires count values
submission['count'] = submission['count'].round().astype(int)

# Save predictions
submission.to_csv('bike_sharing_predictions.csv', index=False)
```

The final steps in the process are:

1. **Apply the model to test data**: Use the trained model to make predictions on the test dataset
2. **Inverse transform predictions**: Convert log-transformed predictions back to the original scale using expm1
3. **Post-process predictions**: Ensure non-negative values and round to integers (since bike counts must be whole numbers)
4. **Create submission file**: Format the predictions according to the competition requirements

## 10. Final Conclusion

Our bike sharing demand prediction model successfully captures the complex patterns that influence bike rental behavior. The model achieves strong performance by leveraging:

1. **Temporal patterns**: The model effectively captures hourly, daily, and seasonal trends using both direct features and cyclic transformations.

2. **Weather impact**: Weather conditions significantly influence bike rental patterns, with temperature being particularly important. The model accounts for both direct weather measurements and derived comfort metrics.

3. **Feature interactions**: The engineered features that combine weather variables help capture complex relationships that affect biking decisions.

4. **Special conditions**: Rush hour flags, weekend patterns, and holiday indicators improve the model's ability to predict demand during different time periods.

The Gradient Boosting Regressor proved to be the most effective algorithm for this task, providing both strong predictive performance and interpretable feature importance.

The final model achieves a low RMSLE on the validation set, indicating that it should perform well on the competition's test data. The predictions account for all the important factors that influence bike sharing demand, making them valuable for capacity planning and resource allocation in bike sharing systems.

This approach demonstrates the power of combining domain knowledge (understanding factors that affect biking behavior) with advanced machine learning techniques to solve real-world prediction problems.
