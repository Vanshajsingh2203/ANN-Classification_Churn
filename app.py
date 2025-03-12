import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model('model.h5')

## Load the encoders and pickle
with open('Label_encoder.pkl','rb') as file:
    Label_encoder= pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streamlit APP
st.title('Customer Churn  Prediction')
 

 ## User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',Label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit_Score')
estimated_salary = st.number_input('Esitemated_Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Nuunber of products',1,4)
has_cr_Card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Menber', [0,1])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [Label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_Card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]    
})

## Onehot encoder 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine onehot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis = 1)

## SCale the input data
input_data_scaled = scaler.transform(input_data)

## Prediction Churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5 :
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')

