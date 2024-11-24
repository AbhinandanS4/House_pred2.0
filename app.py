import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
@st.dialog("We Value Your Feedback")
def feedback_form():
    with st.form(key="feedback_form"):
        name = st.text_input("Name (Optional)")
        email = st.text_input("Email (Optional)")
        feedback = st.text_area("Your Feedback")
        rating = st.slider("Rate our service", 1, 5, 3)
        
            # Submit button
        submitted = st.form_submit_button("Submit Feedback")
        
        # Process form submission
    if submitted:
        st.success("Thank you for your feedback!")
            # Optional: Save the feedback to a file or database
        with open("feedback.txt", "a") as f:
            f.write(f"Name: {name}\nEmail: {email}\nFeedback: {feedback}\nRating: {rating}\n\n")


# Streamlit app UI
st.title("House Price Predictor")


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
st.divider()
st.dataframe(user_data,hide_index=True)
selection = st.selectbox("Select Your Prediction Type", ['Ranged', 'Discrete'])
# Predict Button
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
      

# Create a form
st.divider()
if st.button("Please Add Your Feedback"):
    feedback_form()
   
st.header("FAQs:")
with st.expander("What is Ranged Prediction?"):
    st.write("Ranged Prediction provides a price range estimate, showing both minimum and maximum values for better decision-making.")
with st.expander("What is Discrete Prediction?"):
    st.write("Discrete Prediction provides the exact estimated price. This type of prediction can be in accurate due to multiple factors affecting the outcome in real world situations.")
with st.expander("How do these predictions work?"):
    st.write("This model uses Linear regression which is a statistical method used to model the relationship between a dependent variable (house price) and one or more independent variables (e.g., square footage, number of bedrooms, location). It assumes a linear relationship, meaning the change in price is proportional to the change in the independent variables. By analyzing historical data, the model learns the coefficients (weights) for each independent variable, allowing it to predict the price of a new house based on its features.")
    st.write("To provide a ranged prediction, we can calculate confidence intervals around the point estimate. This interval represents a range of values within which the true house price is likely to fall with a certain level of confidence.")
    st.write("For discrete predictions, we can use techniques like rounding the predicted price to the nearest thousand dollars or assigning it to predefined price categories (e.g., affordable, mid-range, luxury). While linear regression provides a solid foundation for house price prediction, it's important to consider its limitations, such as the assumption of linearity and the potential impact of outliers.")
st.divider()
st.divider()
st.write("Please Rate Our App!!")
st.feedback(options="faces")
