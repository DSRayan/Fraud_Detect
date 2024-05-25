import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the scaler, model, and feature names
scaler = pickle.load(open('scaler_2.pkl', 'rb'))
model = pickle.load(open('random_forest_model_2.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

# Title
st.write("""
# Fraud Detection Model
""")

# User choice for input method
input_method = st.sidebar.radio("Select input method:", ("Upload CSV", "Manual Input"))

if input_method == 'Upload CSV':
    st.header("Upload Transaction Data CSV")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.write(data.head())
        
        # Ensure the uploaded data has the correct features
        try:
            data_features = data[feature_names]
        except KeyError:
            st.error("The uploaded file does not contain the required features.")
        else:
            # Preprocess the data
            data_scaled = scaler.transform(data_features)
            
            # Predict
            if st.button('Predict'):
                predictions = model.predict(data_scaled)
                data['Prediction'] = predictions
                st.write("Prediction summary:")
                count_fraud = data[data['Prediction'] == 1].shape[0]
                count_non_fraud = data[data['Prediction'] == 0].shape[0]
                st.write(f"Number of Fraudulent Transactions: {count_fraud}")
                st.write(f"Number of Legitimate Transactions: {count_non_fraud}")
                st.write("Predictions:")
                st.write(data[['TRANSACTION_ID', 'Prediction']])
                
                st.write("Download the predictions:")
                st.download_button("Download CSV", data.to_csv(index=False), "predicted_transactions.csv")

else:
    st.header("Enter Transaction Data Manually")

    # Create input fields dynamically based on feature names
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.number_input(f'{feature}')

    # Collect user input into a dataframe
    input_data = pd.DataFrame([inputs])
    
    # Preprocess the input
    input_scaled = scaler.transform(input_data)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        # Output the prediction
        if prediction[0] == 1:
            st.write("The transaction is predicted to be FRAUDULENT.")
        else:
            st.write("The transaction is predicted to be LEGITIMATE.")
        
        # Output the prediction probabilities
        st.write("Prediction probabilities (Legitimate, Fraudulent): ", prediction_proba)
