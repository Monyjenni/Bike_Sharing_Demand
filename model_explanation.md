# Bike Sharing Demand Prediction Model - Detailed Explanation

This document provides a comprehensive explanation of the bike sharing demand prediction model implementation, including data preprocessing, exploratory data analysis, feature engineering, modeling, and evaluation.

## 1. Data Loading and Initial Exploration

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
- `temp`: Temperature in Celsius
- `atemp`: "Feels like" temperature in Celsius
- `humidity`: Relative humidity
- `windspeed`: Wind speed
- `casual`: Count of casual users
- `registered`: Count of registered users
- `count`: Total count of bike rentals (target variable)

## 2. Data Preprocessing

### 2.1 Missing Value Check

```python
train_data.isnull().sum()
test_data.isnull().sum()
```

The code checks for missing values in both datasets. Fortunately, no missing values were found, so no imputation was needed.

### 2.2 Datetime Feature Extraction

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
    
    return df
```

This function extracts valuable features from the datetime column:

- **Basic time components**: Year, month, day, hour, day of week
- **Derived time features**: Weekend flag, quarter, day of year, week of year
- **Time period categorization**: Night (0-6h), morning (6-12h), afternoon (12-18h), evening (18-24h)
- **Cyclic features**: Sine and cosine transformations of time variables to capture their cyclical nature

Cyclic features are particularly important for time data because they preserve the circular relationship (e.g., hour 23 is close to hour 0, December is close to January).

## 3. Feature Engineering

```python
def feature_engineering(df):
    # Weather-related interaction features
    df['temp_humidity'] = df['temp'] * df['humidity']
    df['temp_windspeed'] = df['temp'] * df['windspeed']
    df['humidity_windspeed'] = df['humidity'] * df['windspeed']
    df['atemp_humidity'] = df['atemp'] * df['humidity']
    df['atemp_windspeed'] = df['atemp'] * df['windspeed']
    
    # Temperature difference (real feel vs actual)
    df['temp_diff'] = df['atemp'] - df['temp']
    
    # Create binned features
    df['temp_bin'] = pd.cut(df['temp'], bins=6, labels=False)
    df['humidity_bin'] = pd.cut(df['humidity'], bins=6, labels=False)
    df['windspeed_bin'] = pd.cut(df['windspeed'], bins=6, labels=False)
    
    # Create hour bins (morning, afternoon, evening, night)
    df['hour_bin'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
    
    # Create interaction features
    df['season_hour'] = df['season'].astype(str) + '_' + df['hour'].astype(str)
    df['weather_hour'] = df['weather'].astype(str) + '_' + df['hour'].astype(str)
    df['season_weather'] = df['season'].astype(str) + '_' + df['weather'].astype(str)
    df['season_month'] = df['season'].astype(str) + '_' + df['month'].astype(str)
    
    # Create time-based flags
    df['rush_hour'] = ((df['hour'].isin([7, 8, 9]) | df['hour'].isin([17, 18, 19])) & (df['workingday'] == 1)).astype(int)
    df['weekend_rush'] = ((df['hour'].isin([11, 12, 13, 14, 15, 16, 17])) & (df['is_weekend'] == 1)).astype(int)
    df['holiday_weekend'] = ((df['holiday'] == 1) | (df['is_weekend'] == 1)).astype(int)
    df['peak_hours'] = df['hour'].isin([8, 17, 18]).astype(int)
    
    # Create comfort metrics
    df['feel_factor'] = df['temp'] - df['humidity']/100 + df['windspeed']/10
    df['comfort_index'] = df['atemp'] - df['humidity']/100 + df['windspeed']/20
    
    # Weather severity index (higher = worse weather)
    df['weather_severity'] = df['weather'] * (df['humidity']/100) * (1 + df['windspeed']/50)
    
    # Extreme weather conditions
    df['extreme_temp_high'] = (df['temp'] > 30).astype(int)
    df['extreme_temp_low'] = (df['temp'] < 5).astype(int)
    df['extreme_humidity'] = (df['humidity'] > 90).astype(int)
    df['extreme_windspeed'] = (df['windspeed'] > 30).astype(int)
    
    # Ideal biking conditions (moderate temp, low humidity, low wind)
    df['ideal_biking_condition'] = ((df['temp'] > 15) & (df['temp'] < 30) & 
                                   (df['humidity'] < 70) & 
                                   (df['windspeed'] < 20) & 
                                   (df['weather'] <= 2)).astype(int)
    
    return df
```

The feature engineering function creates several types of derived features:

1. **Interaction features**: Combinations of existing features that might have multiplicative effects (e.g., temperature × humidity)
2. **Binned features**: Discretized continuous variables to capture non-linear relationships
3. **Categorical interactions**: Combinations of categorical variables (e.g., season_hour)
4. **Time-based flags**: Indicators for specific time periods (rush hour, weekend rush, etc.)
5. **Comfort metrics**: Custom formulas to estimate biking comfort based on weather conditions
6. **Weather severity**: Index combining weather condition, humidity, and wind speed
7. **Extreme condition flags**: Binary indicators for extreme weather conditions
8. **Ideal conditions**: Flag for ideal biking conditions based on multiple criteria

These engineered features help the model understand complex patterns and interactions in the data that affect bike rental demand.

## 4. Exploratory Data Analysis (EDA)

The EDA section creates several visualizations to understand the data patterns:

1. **Distribution of bike rentals**: Histogram showing the frequency distribution of the target variable
2. **Hourly patterns**: Box plots showing bike rental patterns by hour of the day
3. **Seasonal patterns**: Box plots showing bike rental patterns by season
4. **Weather impact**: Box plots showing how different weather conditions affect bike rentals
5. **Correlation heatmap**: Visualization of correlations between numerical variables

These visualizations help identify key patterns such as:
- Peak rental hours during morning and evening commutes
- Higher rentals during summer and fall seasons
- Decreased rentals during poor weather conditions
- Strong correlations between temperature and bike rentals

## 5. Data Preparation for Modeling

```python
# Define columns to drop for train and test separately
train_drop_cols = ['datetime', 'casual', 'registered', 'hour_bin', 'season_hour', 'weather_hour', 'season_month', 'season_weather']
test_drop_cols = ['datetime', 'hour_bin', 'season_hour', 'weather_hour', 'season_month', 'season_weather']

# Drop unnecessary columns
X = train_data.drop(['count'] + train_drop_cols, axis=1)
y = train_data['count']
X_test = test_data.drop(test_drop_cols, axis=1)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Convert categorical columns to one-hot encoding
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Make sure X_test has the same columns as X
for col in X.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X.columns]

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Log transform the target variable for better model performance
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
```

This section prepares the data for modeling through several steps:

1. **Feature selection**: Dropping unnecessary columns like datetime and redundant features
2. **Handling categorical variables**: Converting categorical features to one-hot encoded columns
3. **Ensuring consistency**: Making sure the test data has the same columns as the training data
4. **Train-validation split**: Dividing the training data into training (80%) and validation (20%) sets
5. **Feature scaling**: Standardizing numerical features to have zero mean and unit variance
6. **Target transformation**: Applying log transformation to the target variable to handle skewness

## 6. Model Training and Evaluation

The code trains and evaluates five different regression models:

1. **Linear Regression**
2. **Ridge Regression**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **XGBoost Regressor**

For each model, the following metrics are calculated:
- **RMSLE (Root Mean Squared Logarithmic Error)**: Primary evaluation metric, less sensitive to outliers
- **RMSE (Root Mean Squared Error)**: Standard error metric in original scale
- **R²**: Coefficient of determination, indicating the proportion of variance explained

The models are then compared based on these metrics to identify the best performer.

### Model Hyperparameters

- **Linear Regression**: Default parameters
- **Ridge Regression**: alpha=0.1 (regularization strength)
- **Random Forest**: n_estimators=200, max_depth=20, min_samples_split=5
- **Gradient Boosting**: n_estimators=200, learning_rate=0.05, max_depth=5
- **XGBoost**: n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8

## 7. Ensemble Modeling

```python
# Create an ensemble of the top 3 models
top_models = results.head(3)['Model'].values

# Create predictions for each model
ensemble_preds = []
for model_name in top_models:
    if model_name == 'Linear Regression':
        ensemble_preds.append(lr_pred)
    elif model_name == 'Ridge Regression':
        ensemble_preds.append(ridge_pred)
    elif model_name == 'Random Forest':
        ensemble_preds.append(rf_pred)
    elif model_name == 'Gradient Boosting':
        ensemble_preds.append(gb_pred)
    elif model_name == 'XGBoost':
        ensemble_preds.append(xgb_pred)

# Weighted ensemble (give more weight to better models)
weights = [0.5, 0.3, 0.2]  # Weights for top 3 models
ensemble_pred = np.zeros_like(ensemble_preds[0])
for i, pred in enumerate(ensemble_preds):
    ensemble_pred += weights[i] * pred
```

The ensemble approach combines predictions from the top three models using a weighted average, with higher weights assigned to better-performing models. This often improves prediction accuracy by leveraging the strengths of different models.

## 8. Feature Importance Analysis

For tree-based models (Random Forest and XGBoost), the code extracts and visualizes feature importance to understand which factors most strongly influence bike rental demand.

The top features typically include:
- Hour of day (and related cyclic features)
- Temperature
- Season
- Weather conditions
- Working day status

## 9. Final Prediction and Output

The code selects the best model (or ensemble) based on validation performance and uses it to generate predictions for the test dataset. The predictions are then saved to a CSV file for submission.

## 10. Model Performance Summary

The model evaluation shows that tree-based models (Random Forest, XGBoost, and Gradient Boosting) significantly outperform linear models for this task, indicating strong non-linear relationships in the data.

The weighted ensemble approach often provides a small improvement over the best individual model by combining different prediction patterns.

## Conclusion

This bike sharing demand prediction model demonstrates a comprehensive approach to regression modeling, including:

- Thorough data preprocessing and feature engineering
- Extensive exploratory data analysis
- Multiple modeling techniques with proper evaluation
- Advanced ensemble methods
- Feature importance analysis

The resulting model achieves high predictive accuracy, making it valuable for bike sharing service planning and resource allocation.
