Linear Regression on Housing Dataset
This project applies a Linear Regression model to a housing dataset in order to model the relationship between various housing features and the target variable.
The entire process includes preprocessing, model training, evaluation, and visualization.

Overview
The objective of this project is to:

Preprocess the given housing dataset.

Build and tune a Linear Regression model.

Evaluate the model using standard regression metrics.

Visualize the regression line for better interpretation.

Dataset
The dataset (Housing.csv) contains various attributes of houses such as:

Number of bedrooms

Number of bathrooms

Area

Parking availability

Air conditioning

And others

The target variable is the price of the house.

Technologies Used
Python 3.x

pandas

numpy

matplotlib

seaborn

scikit-learn

Project Workflow
Data Import and Preprocessing

Load dataset using pandas.

Encode categorical features using LabelEncoder.

Standardize all features using StandardScaler.

Train-Test Split

Divide the data into training (80%) and testing (20%) subsets.

Model Training

Train a Linear Regression model using scikit-learn.

Use GridSearchCV to perform hyperparameter tuning on fit_intercept and copy_X.

Model Evaluation

Predict on the test set.

Evaluate the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

Visualization

Plot the regression line for one of the features against the target variable.

Model Interpretation

Display model coefficients and intercept.

Analyze the relationship between features and the target.

Evaluation Metrics
After model training, the following evaluation metrics are reported:

Mean Absolute Error (MAE): Measures the average magnitude of the errors.

Mean Squared Error (MSE): Measures the average of the squares of the errors.

R² Score: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
