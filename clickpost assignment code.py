import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
# Reading training data, pincodes data (latitude and longitude), and test data using pandas.
# These CSV files contain essential information for model training and prediction.
train_data = pd.read_csv("train_.csv")
pincode_data = pd.read_csv("pincodes.csv")
test_data = pd.read_csv("test_.csv")

# Create dictionaries to map latitude and longitude for each pincode
# The pincodes data provides a mapping of pincodes to geographical coordinates (latitude and longitude).
# This mapping will help enrich the training and test datasets with additional location-based features.
pincode_to_lat = pincode_data.set_index('Pincode')['Latitude'].to_dict()
pincode_to_lon = pincode_data.set_index('Pincode')['Longitude'].to_dict()

# Function to clean, validate, and preprocess the data
# This function performs several operations:
# 1. Maps latitude and longitude for pickup and drop locations using pincodes.
# 2. Extracts year, month, and day components from date columns.
# 3. Removes non-numeric columns (e.g., text data) to ensure compatibility with machine learning models.
# 4. Fills missing values with zeros to avoid errors during model training.
def preprocess_data(data, is_train=True):
    # Map latitude and longitude using pickup and drop pincodes
    data['pickup_latitude'] = data['pickup_pin_code'].map(pincode_to_lat)
    data['pickup_longitude'] = data['pickup_pin_code'].map(pincode_to_lon)
    data['drop_latitude'] = data['drop_pin_code'].map(pincode_to_lat)
    data['drop_longitude'] = data['drop_pin_code'].map(pincode_to_lon)
    
    # Convert date columns to datetime format and extract year, month, and day
    # This allows the model to capture temporal patterns, such as seasonal effects.
    for date_column in ['order_delivered_date', 'order_shipped_date']:
        if date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
            data[f'{date_column}_year'] = data[date_column].dt.year
            data[f'{date_column}_month'] = data[date_column].dt.month
            data[f'{date_column}_day'] = data[date_column].dt.day
            data.drop(columns=[date_column], inplace=True)  # Remove the original date column
    
    # Identify and remove non-numeric columns (e.g., categorical or text data)
    # Non-numeric columns cannot be processed by most machine learning models unless explicitly encoded.
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    data.drop(columns=non_numeric_columns, inplace=True, errors='ignore')
    
    # Fill missing values with zeros
    # This ensures no missing values are present, preventing errors during model training or prediction.
    data.fillna(0, inplace=True)
    
    return data

# Preprocess the training and test datasets using the defined function
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data, is_train=False)

# Ensure the test data has the same columns as the training data
# Machine learning models require the same feature set for both training and prediction.
# Any missing columns in the test data are added with default values (0).
missing_columns = set(train_data.columns) - set(test_data.columns)
for column in missing_columns:
    test_data[column] = 0
# Align the column order in test data with the training data
test_data = test_data[train_data.drop(columns=["order_delivery_sla"]).columns]

# Separate target variable (y) and features (X) from the training dataset
# "order_delivery_sla" is the target variable we aim to predict.
target = train_data["order_delivery_sla"]
features = train_data.drop(columns=["order_delivery_sla"])

# Split the training data into training and validation subsets
# This split (80% training, 20% validation) helps evaluate the model's performance on unseen data.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
# Linear regression is a simple yet effective method for predicting numerical outcomes.
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions on the training and validation subsets
# This evaluates how well the model has learned from the training data and generalizes to new data.
train_predictions = linear_model.predict(X_train)
test_predictions = linear_model.predict(X_test)

# Evaluate the model's performance using MSE and R2 metrics
# MSE (Mean Squared Error) measures the average squared difference between actual and predicted values.
# R2 (R-squared) indicates the proportion of variance in the target variable explained by the model.


# Make predictions on the test dataset
# The trained model is used to predict the SLA (target variable) for the test data.
test_data_predictions = linear_model.predict(test_data)

# Create a submission file with IDs and predicted SLA values
# This file can be submitted for evaluation or further analysis.
submission = pd.DataFrame({'id': test_data['id'], 'predicted_exact_sla': test_data_predictions})
submission.to_csv(r"C:\Users\Asus\OneDrive\Desktop\submission.csv", index=False)
print("Submission file saved successfully!")