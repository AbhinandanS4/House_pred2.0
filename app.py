import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("Housing.csv")
df.drop(columns=['guestroom', 'basement', 'hotwaterheating', 'prefarea'], inplace=True)

# Encoding and scaling
le = LabelEncoder()
scaler = StandardScaler()

# Features and target variable
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

# Encode categorical features
for col in ['mainroad', 'airconditioning', 'furnishingstatus']:
    X_train[col] = le.fit_transform(X_train[col])

# Load model
def load_model():
    with open('lr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Prediction range function
def lr_prediction_range(model, X_new, confidence_level=0.95):
    predictions = model.predict(X_new)
    prediction_std = np.std(model.predict(X_train) - y_train)
    margin_of_error = prediction_std * 1.2  # 10% interval
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

# Streamlit app UI
st.title("House Price Predictor")
selection = st.selectbox("Select Your Prediction Type", ['Ranged', 'Discrete'])

# Collecting user input
st.write("Please Select The Features of Your House")
area = st.text_input("Area:")
bedrooms = st.text_input('Bedrooms:')
bathrooms = st.text_input('Bathrooms:')
stories = st.text_input('Stories:')
parking = st.text_input('Parkings Available:')
mainroad = st.selectbox("Main Road", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning?", ['yes', 'no'])
prefarea=st.selectbox("Preferred Area?",['yes','no'])
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])
# Prepare input data based on user input
user_data = pd.DataFrame({
        'area': [float(area)],
        'bedrooms': [int(bedrooms)],
        'bathrooms': [int(bathrooms)],
        'stories': [int(stories)],
        'mainroad': [mainroad],
        'airconditioning': [airconditioning],
        'parking': [int(parking)],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })
st.dataframe(user_data)

# Predict button
if st.button("Predict"):
    # Encode user input
    user_data['mainroad'] = le.fit_transform(user_data['mainroad'])
    user_data['airconditioning'] = le.fit_transform(user_data['airconditioning'])
    user_data['furnishingstatus'] = le.fit_transform(user_data['furnishingstatus'])
    user_data['prefarea']=le.fit_tranform(user_data['prefarea'])

    # Make prediction
    prediction = model.predict(user_data)[0]
    lower_bound, upper_bound = lr_prediction_range(model, user_data)

    # Display results
    if selection == 'Ranged':
        st.write(f"Your House will cost will be in range: ₹{round(lower_bound[0], 2)} - ₹{round(upper_bound[0], 2)}")
    elif selection == 'Discrete':
        st.write(f"Estimated Cost of the House will be: ₹{round(prediction, 2)}")
st.write("### We Value Your Feedback!")
feedback = st.text_area("Please provide your feedback below:")
if st.button("Submit Feedback"):
    if feedback:
        st.write("Thank you for your feedback!")
        
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n\n")
    else:
        st.write("Please enter your feedback before submitting.")
  
