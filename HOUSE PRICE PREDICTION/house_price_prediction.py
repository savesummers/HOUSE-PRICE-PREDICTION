import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file = pd.read_csv(r"C:\Users\mssam\Desktop\Computer Notes\PYTHON\PROJECTS\HOUSE PRICE PREDICTION\House_price.csv")
X = file[['Avg. Area Income','House Age','Number of Rooms','Number of Bedrooms','Area Population']].values
y = file[['Price']].values

features = ['Avg. Area Income','House Age','Number of Rooms','Number of Bedrooms','Area Population']
for feature in features:
    plt.scatter(file[feature], file['Price'])
    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.show()
#no. of bedrooms not a strong price predictor

model = LinearRegression()
model.fit(X, y)

#make prediction
X_new = list(map(float, input('Enter Features (Avg. Area Income,House Age,Number of Rooms,Number of Bedrooms,Area Population): ').split(',')))
X_new = [X_new]
prediction = model.predict(X_new)
print('House price prediction: ', prediction[0])