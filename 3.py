import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



BASE_DIR=os.path.dirname(__file__)
DATA_PATH=os.path.join(BASE_DIR,"data","Housing.csv")
data=pd.read_csv(DATA_PATH)
# print(data)

# print(data.info())
# print(data.isnull().sum())

le=LabelEncoder()
for i in data.select_dtypes(include="object").columns:
    data[i]=le.fit_transform(data[i])

# print(data)

scaler=StandardScaler()
data[data.columns]=scaler.fit_transform(data[data.columns])
print(data)
x=data.drop(columns="price")
y=data["price"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=GridSearchCV(estimator=LinearRegression(),cv=5,verbose=1,n_jobs=-1,param_grid={
    'fit_intercept':[True,False],
    'copy_X':[True,False]
})
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(pd.DataFrame({"Actual":y_test,"Predicted":y_pred}))
print("Mean Absolute Error:",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))


feature_name = x.columns[0]

plt.figure(figsize=(10,6))
sns.scatterplot(x=x_test[feature_name], y=y_test, label='Actual', color='blue')
sns.lineplot(x=x_test[feature_name], y=y_pred, label='Predicted', color='red')
plt.title(f"Regression Line for {feature_name} vs Price")
plt.xlabel(feature_name)
plt.ylabel('Price')
plt.legend()
plt.show()
