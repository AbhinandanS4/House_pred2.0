import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Housing.csv")
# Drop any features you do not want to use in the model
df.drop(columns=['guestroom', 'basement', 'hotwaterheating'], inplace=True)

# Initialize LabelEncoder and fit it on the full dataset
le_mainroad = LabelEncoder()
le_airconditioning = LabelEncoder()
le_furnishingstatus = LabelEncoder()
le_prefarea = LabelEncoder()

# Encode categorical features in the full dataset
df['mainroad'] = le_mainroad.fit_transform(df['mainroad'])
df['airconditioning'] = le_airconditioning.fit_transform(df['airconditioning'])
df['furnishingstatus'] = le_furnishingstatus.fit_transform(df['furnishingstatus'])
df['prefarea'] = le_prefarea.fit_transform(df['prefarea'])

# Features and target variable
X = df.drop(columns=['price'])
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

# Load model
def load_model():
    with open('lr_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Prediction range function
def lr_prediction_range(model, X_new):
    predictions = model.predict(X_new)
    prediction_std = np.std(model.predict(X_train) - y_train)
    margin_of_error = prediction_std * 1.2  # 10% interval
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound
def format_indian_number(num):
    num_str = str(int(num))
    if len(num_str) > 3:
        last_three = num_str[-3:]
        other_digits = num_str[:-3]
        formatted_number = ""
        while len(other_digits) > 2:
            formatted_number = ',' + other_digits[-2:] + formatted_number
            other_digits = other_digits[:-2]
        formatted_number = other_digits + formatted_number
        return formatted_number + ',' + last_three
    return num_str
# Streamlit app UI
st.title("House Price Predictor")
selection = st.selectbox("Select Your Prediction Type", ['Ranged', 'Discrete'])

# Collecting user input
st.write("Please Select The Features of Your House")
area = st.slider("Area (in sq ft.):", 100, 17000, 5000)
bedrooms = st.slider('Bedrooms:', 0, 15, 10)
bathrooms = st.slider('Bathrooms:', 0, 15, 10)
stories = st.slider('Stories:', 0, 10, 5)
parking = st.slider('Parkings Available:', 0, 5, 2)
mainroad = st.selectbox("Main Road", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning?", ['yes', 'no'])
prefarea = st.selectbox("Preferred Area?", ['yes', 'no'])
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# Prepare input data based on user input
user_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

st.dataframe(user_data)

# Predict button
if st.button("Predict"):
    # Encode user input using the same label encoders
    user_data['mainroad'] = le_mainroad.transform(user_data['mainroad'])
    user_data['airconditioning'] = le_airconditioning.transform(user_data['airconditioning'])
    user_data['furnishingstatus'] = le_furnishingstatus.transform(user_data['furnishingstatus'])
    user_data['prefarea'] = le_prefarea.transform(user_data['prefarea'])

    # Make prediction
    prediction = model.predict(user_data)[0]
    lower_bound, upper_bound = lr_prediction_range(model, user_data)

    # Display results
    if selection == 'Ranged':
        st.write(f"Your House will cost in range: ₹{format_indian_number(round(lower_bound[0], 2))} - ₹{format_indian_number(round(upper_bound[0], 2))}")
    elif selection == 'Discrete':
        st.write(f"Estimated Cost of the House will be: ₹{format_indian_number(round(prediction, 2))}")


# Feedback section
st.write("### We Value Your Feedback!")
feedback = st.text_area("Please provide your feedback below:")
if st.button("Submit Feedback"):
    if feedback:
        st.write("Thank you for your feedback!")
        with open("feedback.txt", "a") as f:
            f.write(feedback + "\n\n")
    else:
        st.write("Please enter your feedback before submitting.")
