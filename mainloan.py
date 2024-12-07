import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


st.title("Loan Prediction")
st.text_input("enter your name")
lp1=st.number_input("Enter your age",0,100)
lp2=st.selectbox("Gender",["Male","Female"])
lp3=st.selectbox("Education",["Master","Bachelor","High School","Associate","Doctorate"])
lp4=st.selectbox("income",["0-100000","100000-300000","400000-800000","800000-1200000","1200000-2000000","above 20 lpa"])
lp5=st.number_input("enter your Employment Experience",0,100)
lp6=st.selectbox("Home ownership",["own","Rent","MORTGAGE","other"])
lp7=st.number_input("Enter the loan amount you are applying for",500,50000000)
lp8=st.selectbox("Loan intent",["Person", "Education", 'Medical', 'Venture', 'Homeimprovement',
       'Debtconsolidation'])
lp9=st.number_input("Enter the loan interest rate",0.00,50.00)
lp10=st.number_input("Enter your monthly loan replayment",0.00,100.00)
lp11=st.number_input("Enter the length of your credit history (in years) ",0.00,40.00)
lp12=st.number_input("Enter your credit_score",300,850)
lp13=st.selectbox("Have you had any previous loan defaults or missed payments?", ["No", "Yes"])


gender = enc.fit_transform([lp2])[0]
education = enc.fit_transform([lp3])[0]
income = enc.fit_transform([lp4])[0]
homeownership = enc.fit_transform([lp6])[0]
loan_intent = enc.fit_transform([lp8])[0]
previous_defaults = enc.fit_transform([lp13])[0]


with open('rf.pkl','rb') as f:
    model=pickle.load(f)
    f.close()
if st.button('predict') :
    data=np.array([[lp1, gender, education, income, lp5, homeownership, lp7, loan_intent, lp9, lp10, lp11, lp12, previous_defaults]])
    prediction =model.predict(data)[0]

    if prediction == 1:
        st.write("congratulations your loan got approved")
    elif prediction == 0:
        st.write("Sorry your loan got rejected")
