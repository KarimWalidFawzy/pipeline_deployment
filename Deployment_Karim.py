import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
from sklearn import preprocessing

import pyngrok
import streamlit_ace
rawdata=open("combined_pipeline.pkl")
cols=[]    
def main(): 
    st.title("")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:black;text-align:center;">Model Deployment App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
# Load the pickled model
logistic_model, linear_model, selected_features_list = joblib.load(filename="combined_pipeline.pkl")

# Function to make predictions
def make_predictions(input_data):
    input_df = pd.DataFrame([input_data], columns=selected_features_list)
    everdelinquent_pred = logistic_model.predict(input_df)[0]
    prepayment_pred = linear_model.predict(input_df)[0]
    
    # Convert everdelinquent prediction to textual representation
    everdelinquent_text = 'Loan is EverDelinquent' if everdelinquent_pred == 1 else 'Loan is Non-Delinquent'
    
    # Format prepayment prediction to include USD and round to two decimal places
    prepayment_pred = f"${prepayment_pred:.2f}"
    
    return everdelinquent_text, prepayment_pred

# Streamlit app

st.header(":wave: :violet[**Welcome All!**]", divider='rainbow')

st.title('🏡 :rainbow[Mortgage Prepayment Prediction App]')

# Project description
st.markdown("""
<style>
    .description {
        font-size: 18px;
        color: #4B4B4B;
    }
    .prediction {
        font-size: 20px;
        color: #008000;
    }
    .thank-you{
        font-size: 18px;
        color: #0000ff;
            }    
</style>
<div class="description">
    Welcome to the Loan Status Prediction App! This application helps predict the likelihood of a loan becoming delinquent and estimates the prepayment amount based on various features of the loan and borrower. 
    Simply enter the required details, and the app will provide you with the predictions.
</div>
""", unsafe_allow_html=True)

# Create input fields for each feature
input_data = {}
for feature in selected_features_list:
    input_data[feature] = st.text_input(f'Enter {feature}')

# Button to make predictions
if st.button('Predict'):
    everdelinquent_pred, prepayment_pred = make_predictions(input_data)
    st.markdown(f'<div class="prediction">EverDelinquent Status: {everdelinquent_pred}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Prepayment Amount: {prepayment_pred}</div>', unsafe_allow_html=True)
    st.markdown('<div class="thank-you">Thank you for using our prediction service!</div>', unsafe_allow_html=True)

    
      
if __name__=='__main__': 
    main()
