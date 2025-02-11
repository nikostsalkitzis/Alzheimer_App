import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('best_model.pkl')

# Function to make predictions
def predict(data):
    prediction = model.predict(data)
    return prediction

# Streamlit frontend
st.set_page_config(page_title="Alzheimer's Disease Prediction", page_icon="🧠", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #262730;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4a148c;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4a148c;
    }
    .stMarkdown p {
        font-size: 16px;
    }
    .stSidebar {
        background-color: #e0bbff;
        padding: 20px;
        border-radius: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Title and description
st.title("Alzheimer's Disease Prediction 🧠")
st.write("""
This app helps predict whether a person has Alzheimer's disease based on certain lifestyle and health factors. 
Fill in the details below to get your prediction.
""")

# Sidebar for additional information
with st.sidebar:
    st.header("About Alzheimer's Disease")
    st.write("""
    Alzheimer's disease is a progressive disorder that causes brain cells to waste away (degenerate) and die. 
    It's the most common cause of dementia — a continuous decline in thinking, behavioral and social skills 
    that disrupts a person's ability to function independently.
    """)
    st.write("**Symptoms:**")
    st.write("- Memory loss")
    st.write("- Difficulty in thinking and reasoning")
    st.write("- Difficulty in making judgments and decisions")
    st.write("- Changes in personality and behavior")
    st.write("**Risk Factors:**")
    st.write("- Age")
    st.write("- Family history")
    st.write("- Genetics")
    st.write("- Head trauma")
    st.write("- Lifestyle and heart health")
    st.write("**Prevention Tips:**")
    st.write("- Regular physical activity")
    st.write("- Healthy diet")
    st.write("- Mental stimulation")
    st.write("- Quality sleep")
    st.write("- Social engagement")

# Organize input fields in columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👵 Age", min_value=0, max_value=120, value=60)
    gender = st.selectbox("👤 Gender", ['Male', 'Female'])
    education_level = st.selectbox("🎓 Education Level", ['Low', 'Medium', 'High'])
    bmi = st.number_input("🏋️‍♀️ BMI", min_value=0.0, value=25.0)
    physical_activity = st.selectbox("🏃‍♂️ Physical Activity Level", ['Low', 'Medium', 'High'])
    smoking_status = st.selectbox("🚬 Smoking Status", ['Current', 'Former', 'Never'])
    alcohol_consumption = st.selectbox("🍷 Alcohol Consumption", ['Never', 'Regularly', 'Occasionally'])
    diabetes = st.selectbox("💉 Diabetes", ['No', 'Yes'])
    hypertension = st.selectbox("❤️ Hypertension", ['No', 'Yes'])
    marital_status = st.selectbox("💍 Marital Status", ['Widowed', 'Single', 'Married'])
    genetic_risk = st.selectbox("🧬 Genetic Risk (APOE-ε4 allele)", ['No', 'Yes'])
    social_engagement = st.selectbox("🗣  Social Engagement Level", ['Low', 'Medium', 'High'])

with col2:
    cholesterol_level = st.selectbox("💪 Cholesterol Level", ['Normal', 'High'])
    family_history = st.selectbox("👪 Family History of Alzheimer’s", ['No', 'Yes'])
    cognitive_test_score = st.number_input("📝 Cognitive Test Score", min_value=0, max_value=30, value=15)
    depression_level = st.selectbox("😞 Depression Level", ['Low', 'Medium', 'High'])
    sleep_quality = st.selectbox("🛏 Sleep Quality", ['Poor', 'Average', 'Good'])
    dietary_habits = st.selectbox("🍽 Dietary Habits", ['Unhealthy', 'Average', 'Healthy'])
    air_pollution = st.selectbox("🌫 Air Pollution Exposure", ['Low', 'Medium', 'High'])
    employment_status = st.selectbox("💼 Employment Status", ['Unemployed', 'Retired', 'Employed'])
    income_level = st.selectbox("💰 Income Level", ['Low', 'Medium', 'High'])
    stress_level = st.selectbox("😰 Stress Levels", ['Low', 'Medium', 'High'])
    urban_rural = st.selectbox("🌆 Urban vs Rural Living", ['Rural', 'Urban'])


# Create a DataFrame for input
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education_level],
    'BMI': [bmi],
    'Physical Activity Level': [physical_activity],
    'Smoking Status': [smoking_status],
    'Alcohol Consumption': [alcohol_consumption],
    'Diabetes': [diabetes],
    'Hypertension': [hypertension],
    'Cholesterol Level': [cholesterol_level],
    'Family History of Alzheimer’s': [family_history],
    'Cognitive Test Score': [cognitive_test_score],
    'Depression Level': [depression_level],
    'Sleep Quality': [sleep_quality],
    'Dietary Habits': [dietary_habits],
    'Air Pollution Exposure': [air_pollution],
    'Employment Status': [employment_status],
    'Marital Status': [marital_status],
    'Genetic Risk Factor (APOE-ε4 allele)': [genetic_risk],
    'Social Engagement Level': [social_engagement],
    'Income Level': [income_level],
    'Stress Levels': [stress_level],
    'Urban vs Rural Living': [urban_rural]
})

# Preprocess the input (map categorical values to numbers as done earlier)
family_history_mapping = {'No': 0, 'Yes': 1}
activity_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
smoking_status_mapping = {'Current': 0, 'Former': 1, 'Never': 2}
alcohol_consumption_mapping = {'Never': 0, 'Regularly': 1, 'Occasionally': 2}
diabetes_mapping = {'No': 0, 'Yes': 1}
hypertension_mapping = {'No': 0, 'Yes': 1}
cholesterol_level_mapping = {'Normal': 0, 'High': 1}
depression_level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
sleep_quality_mapping = {'Poor': 0, 'Average': 1, 'Good': 2}
dietary_habits_mapping = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
air_pollution_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
employment_status_mapping = {'Unemployed': 0, 'Retired': 1, 'Employed': 2}
marital_status_mapping = {'Widowed': 0, 'Single': 1, 'Married': 2}
genetic_risk_mapping = {'No': 0, 'Yes': 1}
social_engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
income_level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
stress_level_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
urban_mapping = {'Rural': 0, 'Urban': 1}
gender_mapping = {'Male': 0, 'Female': 1}  # Add this mapping for gender

# Apply the mappings
input_data['Gender'] = input_data['Gender'].map(gender_mapping)
input_data['Physical Activity Level'] = input_data['Physical Activity Level'].map(activity_mapping)
input_data['Smoking Status'] = input_data['Smoking Status'].map(smoking_status_mapping)
input_data['Alcohol Consumption'] = input_data['Alcohol Consumption'].map(alcohol_consumption_mapping)
input_data['Diabetes'] = input_data['Diabetes'].map(diabetes_mapping)
input_data['Hypertension'] = input_data['Hypertension'].map(hypertension_mapping)
input_data['Cholesterol Level'] = input_data['Cholesterol Level'].map(cholesterol_level_mapping)
input_data['Depression Level'] = input_data['Depression Level'].map(depression_level_mapping)
input_data['Sleep Quality'] = input_data['Sleep Quality'].map(sleep_quality_mapping)
input_data['Dietary Habits'] = input_data['Dietary Habits'].map(dietary_habits_mapping)
input_data['Air Pollution Exposure'] = input_data['Air Pollution Exposure'].map(air_pollution_mapping)
input_data['Employment Status'] = input_data['Employment Status'].map(employment_status_mapping)
input_data['Marital Status'] = input_data['Marital Status'].map(marital_status_mapping)
input_data['Genetic Risk Factor (APOE-ε4 allele)'] = input_data['Genetic Risk Factor (APOE-ε4 allele)'].map(genetic_risk_mapping)
input_data['Social Engagement Level'] = input_data['Social Engagement Level'].map(social_engagement_mapping)
input_data['Income Level'] = input_data['Income Level'].map(income_level_mapping)
input_data['Stress Levels'] = input_data['Stress Levels'].map(stress_level_mapping)
input_data['Urban vs Rural Living'] = input_data['Urban vs Rural Living'].map(urban_mapping)
input_data['Family History of Alzheimer’s'] = input_data['Family History of Alzheimer’s'].map(family_history_mapping)
input_data['Education Level'] = input_data['Education Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
input_data['BMI'] = input_data['BMI']  # BMI is already numerical, so no mapping needed
input_data['Cognitive Test Score'] = input_data['Cognitive Test Score']  # No mapping needed

# Make the prediction
if st.button("Predict"):
    prediction = predict(input_data)

    # Display the result
    st.write("### Input data:")
    st.write(input_data)

    st.write("### Prediction result:")
    if prediction == 1:
        st.markdown("🔴 **The model predicts that the person may have Alzheimer's disease.**")
    else:
        st.markdown("🟢 **The model predicts that the person may not have Alzheimer's disease.**")

# Add footer text
st.write("### Disclaimer:")
st.write("This app is a tool for predicting Alzheimer's disease based on lifestyle factors and health metrics. It is not a substitute for a medical diagnosis.")

st.markdown(
    """
    <div style="text-align: center; padding: 15px; background-color: #e0bbff; border-radius: 10px;">
        <h4 style="color: #4a148c;">👩‍🔬 Authors 👨‍🔬</h4>
        <p style="font-size:18px; color: #000;">
            📌 <strong>Konstantina Koulaktsidou</strong> <br>
            📌 <strong>Nikolaos Lappas</strong> <br>
            📌 <strong>Aikaterini Rousounelou</strong> <br>
            📌 <strong>Nikolaos Tsalkitzis</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
