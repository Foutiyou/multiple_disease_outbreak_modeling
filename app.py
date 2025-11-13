
import streamlit as st
import pandas as pd
import joblib

# Load the preprocessor
preprocessor = joblib.load('preprocessor.joblib')

# Load all trained models
models = {
    'Meningite_Outbreak': joblib.load('random_forest_model_meningite_outbreak.joblib'),
    'Rougeole_Outbreak': joblib.load('random_forest_model_rougeole_outbreak.joblib'),
    'Dengue_Cas_confirme_(hebdomadaire)_Outbreak': joblib.load('random_forest_model_dengue_cas_confirme_(hebdomadaire)_outbreak.joblib'),
    'Cholera_Outbreak': joblib.load('random_forest_model_cholera_outbreak.joblib'),
    'Covid19_Outbreak': joblib.load('random_forest_model_covid19_outbreak.joblib'),
}

st.title('Epidemic Outbreak Prediction Dashboard')
st.write('Predict if an epidemic outbreak will occur for a selected disease based on input features.')

# --- Input features from user ---
st.header('Input Data for Prediction')

selected_disease = st.selectbox(
    'Select Disease for Prediction:',
    list(models.keys())
)

organisationunitname = st.selectbox(
    'Select Organisation Unit:',
    ('Bakel', 'Dakar Centre', 'Dakar Nord', 'Dakar Ouest', 'Dakar Sud', 'Diamniadio', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 'Kedougou', 'Kolda', 'Louga', 'Matam', 'Mbour', 'Pikine', 'Podor', 'Rufisque', 'Saint-Louis', 'Sedhiou', 'Tambacounda', 'Thies', 'Ziguinchor') # Example list of units
)

week_of_year = st.slider('Week of Year (1-52):', 1, 52, 1)
day_of_week = st.slider('Day of Week (0=Monday, 6=Sunday):', 0, 6, 0)

# Initialize all lagged features to 0
meningite_lag1, meningite_lag2, meningite_lag3 = 0, 0, 0
rougeole_lag1, rougeole_lag2, rougeole_lag3 = 0, 0, 0
dengue_lag1, dengue_lag2, dengue_lag3 = 0, 0, 0
cholera_lag1, cholera_lag2, cholera_lag3 = 0, 0, 0
covid19_lag1, covid19_lag2, covid19_lag3 = 0, 0, 0

st.subheader('Previous Week Confirmed Cases for Selected Disease:')

# Display lagged features only for the selected disease
if selected_disease == 'Meningite_Outbreak':
    meningite_lag1 = st.number_input('Meningitis (Lag 1):', min_value=0, value=0)
    meningite_lag2 = st.number_input('Meningitis (Lag 2):', min_value=0, value=0)
    meningite_lag3 = st.number_input('Meningitis (Lag 3):', min_value=0, value=0)
elif selected_disease == 'Rougeole_Outbreak':
    rougeole_lag1 = st.number_input('Measles (Lag 1):', min_value=0, value=0)
    rougeole_lag2 = st.number_input('Measles (Lag 2):', min_value=0, value=0)
    rougeole_lag3 = st.number_input('Measles (Lag 3):', min_value=0, value=0)
elif selected_disease == 'Dengue_Cas_confirme_(hebdomadaire)_Outbreak':
    dengue_lag1 = st.number_input('Dengue (Lag 1):', min_value=0, value=0)
    dengue_lag2 = st.number_input('Dengue (Lag 2):', min_value=0, value=0)
    dengue_lag3 = st.number_input('Dengue (Lag 3):', min_value=0, value=0)
elif selected_disease == 'Cholera_Outbreak':
    cholera_lag1 = st.number_input('Cholera (Lag 1):', min_value=0, value=0)
    cholera_lag2 = st.number_input('Cholera (Lag 2):', min_value=0, value=0)
    cholera_lag3 = st.number_input('Cholera (Lag 3):', min_value=0, value=0)
elif selected_disease == 'Covid19_Outbreak':
    covid19_lag1 = st.number_input('Covid-19 (Lag 1):', min_value=0, value=0)
    covid19_lag2 = st.number_input('Covid-19 (Lag 2):', min_value=0, value=0)
    covid19_lag3 = st.number_input('Covid-19 (Lag 3):', min_value=0, value=0)


if st.button('Predict Outbreak'):
    # Get the selected model
    rf_model = models[selected_disease]

    input_data = pd.DataFrame([{
        'organisationunitname': organisationunitname,
        'week_of_year': week_of_year,
        'day_of_week': day_of_week,
        'Meningite_lag1': meningite_lag1,
        'Meningite_lag2': meningite_lag2,
        'Meningite_lag3': meningite_lag3,
        'Rougeole_lag1': rougeole_lag1,
        'Rougeole_lag2': rougeole_lag2,
        'Rougeole_lag3': rougeole_lag3,
        'Dengue_Cas_confirme_(hebdomadaire)_lag1': dengue_lag1,
        'Dengue_Cas_confirme_(hebdomadaire)_lag2': dengue_lag2,
        'Dengue_Cas_confirme_(hebdomadaire)_lag3': dengue_lag3,
        'Cholera_lag1': cholera_lag1,
        'Cholera_lag2': cholera_lag2,
        'Cholera_lag3': cholera_lag3,
        'Covid19_lag1': covid19_lag1,
        'Covid19_lag2': covid19_lag2,
        'Covid19_lag3': covid19_lag3
    }])

    # Preprocess the input data
    processed_input = preprocessor.transform(input_data)

    # Make prediction
    prediction = rf_model.predict(processed_input)

    # Check if the positive class (1) exists in the model's classes
    if 1 in rf_model.classes_:
        pos_class_idx = list(rf_model.classes_).index(1)
        prediction_proba = rf_model.predict_proba(processed_input)[:, pos_class_idx]
    else:
        # If the model was not trained with class 1, then the probability of class 1 is 0
        prediction_proba = [0.0]

    st.subheader(f'Prediction Result for {selected_disease.replace("_Outbreak", "").replace("_", " ")}:')
    if prediction[0] == 1:
        st.error(f'Likely to have an Outbreak! (Probability: {prediction_proba[0]:.2f})')
    else:
        st.success(f'No Outbreak predicted. (Probability of outbreak: {prediction_proba[0]:.2f})')
