import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

import os
os.chdir(r'D:\Trainings\Bodhayan\DS Suite\Linear_Regression_MultiVariate')

df = pd.read_csv('homeprices.csv')
print(df)

#**Data Preprocessing: Fill NA values with median value of a column**
print(df.bedrooms.median())

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
print(df)
# Split into training and testing datasets (80% train, 20% test)
# Split data into features (X) and target (y)
X = df.drop('price', axis='columns')
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

#Coeffecients
print(reg.coef_)
print(reg.intercept_)



#Make predictions on the dataset
y_pred = reg.predict(X_test)
# Calculate Mean Squared Error on test data
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Data:", mse)

#Calculate the RMSE on the dataset
rmse = np.sqrt(mse)
print("Root Mean Squared Error on Test Data:", rmse)
#Comapre predicted/actuals
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual vs Predicted:\n", comparison)

#Lets save the Model into a pickle file
with open('model_pickle','wb') as fp:
    pickle.dump(reg,fp)

#Lets open the model for prediction

'''with open('model_pickle','rb') as f:
    mp = pickle.load(f)
#Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old  
print(f'Predictions are ${mp.predict([[3000, 3, 40]])}')'''