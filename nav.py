import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df=pd.read_csv(r'Salary_Data.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

accuracy=regressor.score(x_test,y_test)
st.title('Salary Prediction')
experience=st.number_input('Enter years of experience')
if st.button('Predict Salary'):
    predicted_salary=regressor.predict([[experience]])
    st.write(f'Predicted Salary: ${predicted_salary[0]:.2f}')
