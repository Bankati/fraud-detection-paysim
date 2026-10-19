import streamlit as st
import pandas as pd
import joblib


#Import de notre model de machine learning
model=joblib.load("fraud_detection_pipeline.pkl")

#Création de notre interface utilisateur
#Titre de l'application
st.title("Détection apps")

#Création de nos champs de saisie
st.markdown("please enter the transaction and use the predict button")

st.divider()

transaction_type=st.selectbox("Transaction type",["CASH_OUT","PAYMENT","CASH_IN","TRANSFER","DEBIT"])
amount=st.number_input("Amount",min_value=0.0,value=1000.0)

newbalanceOrig=st.number_input("New balance value",min_value=0.0,value=1000.0,key="newbalanceOrig")
oldbalanceOrg=st.number_input("old balance(recipient)",min_value=0.0,value=1000.0,key="oldbalanceOrg")
oldbalanceDest=st.number_input("New balance value",min_value=0.0,value=1000.0,key="oldbalanceDest")
newbalanceDest=st.number_input("New balance value",min_value=0.0,value=100.0,key="newbalanceDest")



#'type' 'amount' 'oldbalanceOrg' 'newbalanceOrig' 'oldbalanceDest''newbalanceDest'

#Création d'un bouton de prédiction
if st.button("Predict"):
    input_data=pd.DataFrame({
        "type":[transaction_type],
        "amount":[amount],
        "oldbalanceOrg":[oldbalanceOrg],
        "newbalanceOrig":[newbalanceOrig],
        "oldbalanceDest":[oldbalanceDest],
        "newbalanceDest":[newbalanceDest]   
    })

    #Prediciton
    prediction=model.predict(input_data)[0]

    st.subheader(f"Prediction: {prediction}")

    if prediction==1:
        st.error("This transaction is predicted to be fraudulent.")
    else:
        st.success("This transaction is predicted to be non-fraudulent.")