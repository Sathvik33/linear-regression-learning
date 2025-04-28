# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Set up the base directory and data path
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "Housing.csv")

# Load the dataset
data = pd.read_csv(DATA_PATH)
print(data)

# Display information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Encode categorical variables using Label Encoding
le = LabelEncoder()
for i in data.select_dtypes(include="object").columns:
    data[i] = le.fit_transform(data[i])

print(data)

# Standardize the dataset
scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])
print(data)

# Separate features (x) and target variable (y)
x = data.drop(columns="price")
y = data["price"]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize GridSearchCV with Linear Regression and perform hyperparameter tuning
model = GridSearchCV(
    estimator=LinearRegression(),
    cv=5,
    verbose=1,
    n_jobs=-1,
    param_grid={
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
)

# Train the model on the training data
model.fit(x_train, y_train)

# Predict the target variable for the test data
y_pred = model.predict(x_test)

# Display actual vs predicted values
print(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))

# Evaluate the model using different metrics
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualization: Plot actual vs predicted values for the first feature
feature_name = x.columns[0]  # Select the first feature for plotting

plt.figure(figsize=(10,6))
sns.scatterplot(x=x_test[feature_name], y=y_test, label='Actual', color='blue')
sns.lineplot(x=x_test[feature_name], y=y_pred, label='Predicted', color='red')
plt.title(f"Regression Line for {feature_name} vs Price")
plt.xlabel(feature_name)
plt.ylabel('Price')
plt.legend()
plt.show()

# Print model coefficients and intercept
print("Model Coefficients:", model.best_estimator_.coef_)
print("Model Intercept:", model.best_estimator_.intercept_)
