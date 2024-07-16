import pandas as pd
dataset = pd.read_csv('training_data.csv')
dataset.dropna(inplace=True)
dataset = dataset.drop(columns=['production '], axis=1)
y = dataset['recovery']

x = dataset.drop(columns=['recovery'],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

y_model_train_pred = model.predict(x_train)
y_model_test_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
model_train_mse = mean_squared_error(y_train, y_model_train_pred)
model_train_r2 = r2_score(y_train, y_model_train_pred)
model_test_mse = mean_squared_error(y_test, y_model_test_pred)
model_test_r2 = r2_score(y_test, y_model_test_pred)

print(model_train_mse)
print(model_train_r2)

import pickle
pickle.dump(model, open('trainedModel.pkl','wb'))

