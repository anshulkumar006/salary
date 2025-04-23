import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('salaries.csv')

st.title('Salary Prediction App')
st.write("Dataset Preview:")
st.write(data.head())

X = data[['YearsExperience']] 
y = data['Salary']

model = LinearRegression()
model.fit(X, y)

experience = st.number_input('Enter Years of Experience:', min_value=0.0, max_value=50.0, step=0.1)

if st.button('Predict Salary'):
    prediction = model.predict([[experience]])
    st.success(f'Estimated Salary: ${prediction[0]:,.2f}')
