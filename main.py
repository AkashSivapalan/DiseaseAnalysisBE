import os
from fastapi import FastAPI
from dotenv import load_dotenv
from  openai import OpenAI
import re
import joblib
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd


load_dotenv()

app = FastAPI()

OPENAI_KEY = os.getenv("OPENAI_KEY")

app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=OPENAI_KEY,
)

model = joblib.load('disease_prediction_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Symptoms extracted from the dataset
symptoms_list = ['stomach_pain', 'toxic_look_(typhos)', 'continuous_sneezing', 'ulcers_on_tongue', 'family_history', 'weight_gain', 'joint_pain', 'mood_swings', 'blood_in_sputum', 'belly_pain', 'weakness_in_limbs', 'fluid_overload', 'burning_micturition', 'obesity', 'fast_heart_rate', 'anxiety', 'distention_of_abdomen', 'depression', 'swollen_blood_vessels', 'skin_peeling', 'cough', 'prominent_veins_on_calf', 'dark_urine', 'receiving_unsterile_injections', 'pain_during_bowel_movements', 'throat_irritation', 'swelled_lymph_nodes', 'muscle_pain', 'stomach_bleeding', 'palpitations', 'restlessness', 'slurred_speech', 'loss_of_balance', 'rusty_sputum', 'hip_joint_pain', 'skin_rash', 'stiff_neck', 'small_dents_in_nails', 'foul_smell_of urine', 'dischromic _patches', 'unsteadiness', 'back_pain', 'nodal_skin_eruptions', 'movement_stiffness', 'sweating', 'swelling_of_stomach', 'passage_of_gases', 'weight_loss', 'receiving_blood_transfusion', 'polyuria', 'acidity', 'spinning_movements', 'irritability', 'congestion', 'vomiting', 'puffy_face_and_eyes', 'blackheads', 'patches_in_throat', 'redness_of_eyes', 'history_of_alcohol_consumption', 'breathlessness', 'yellow_crust_ooze', 'weakness_of_one_body_side', 'excessive_hunger', 'swelling_joints', 'bruising', 'swollen_extremeties', 'dehydration', 'visual_disturbances', 'watering_from_eyes', 'swollen_legs', 'altered_sensorium', 'bladder_discomfort', 'cramps', 'indigestion', 'dizziness', 'loss_of_smell', 'red_spots_over_body', 'yellowing_of_eyes', 'neck_pain', 'lack_of_concentration', 'internal_itching', 'fatigue', 'spotting_ urination', 'malaise', 'runny_nose', 'drying_and_tingling_lips', 'coma', 'yellowish_skin', 'phlegm', 'diarrhoea', 'enlarged_thyroid', 'bloody_stool', 'extra_marital_contacts', 'chest_pain', 'brittle_nails', 'painful_walking', 'chills', 'itching', 'irritation_in_anus', 'pain_behind_the_eyes', 'irregular_sugar_level', 'abnormal_menstruation', 'scurring', 'abdominal_pain', 'mild_fever', 'pus_filled_pimples', 'silver_like_dusting', 'yellow_urine', 'muscle_wasting', 'constipation', 'nausea', 'knee_pain', 'acute_liver_failure', 'continuous_feel_of_urine', 'inflammatory_nails', 'mucoid_sputum', 'sunken_eyes', 'lethargy', 'blister', 'cold_hands_and_feets', 'blurred_and_distorted_vision', 'muscle_weakness', 'high_fever', 'headache', 'increased_appetite', 'red_sore_around_nose', 'loss_of_appetite', 'pain_in_anal_region', 'sinus_pressure', 'shivering']

df = pd.read_csv('Disease precaution.csv')
# Ensure that all precaution columns are treated as strings, and fill NaN with an empty string
df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']] = df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].fillna('').astype(str)

# Combine the four precaution columns into a single string, separated by commas
df['Precautions'] = df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].apply(lambda x: ', '.join(filter(None, x)), axis=1)


def check_symptoms(message):
    prompt = f"User described: '{message}'. Identify any of the following symptoms that match: {', '.join(symptoms_list)}. Please return only the matching symptoms with the same value I listed as a comma-separated list."
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract the response
    matched_symptoms_text = response.choices[0].message.content.strip()
    
    # Filter the matched symptoms by checking each symptom in the symptoms_list
    matched_symptoms = []
    
    # Convert message to lowercase for case-insensitive matching
    message_lower = matched_symptoms_text.lower()
    
    # Check if any symptom in symptoms_list is present in the message
    for symptom in symptoms_list:
        if re.search(r'\b' + re.escape(symptom) + r'\b', message_lower):
            matched_symptoms.append(symptom)
    
    return matched_symptoms

def predictDiseases(matched_symptoms):
    symptom_str = ' '.join(matched_symptoms)
    X_new = vectorizer.transform([symptom_str])
    # Predict the disease
    predicted_disease_encoded = model.predict(X_new)
    predicted_disease = label_encoder.inverse_transform(predicted_disease_encoded)

    return predicted_disease


def get_precautions(disease_name):
    # Filter for the specific disease
    disease_info = df[df['Disease'] == disease_name]
    
    # If the disease is found, return the precautions
    if not disease_info.empty:
        precautions = disease_info.iloc[0]['Precautions']
        return precautions
    else:
        return ""

class Message(BaseModel):
    message: str

@app.post("/diseasePrediction")
def recommendations(userMessage:Message ):
    matched_symptoms = check_symptoms(userMessage.message)

    # If no symptoms found, then return message without disease prediction
    if len(matched_symptoms)==0:
        return {"message": "Sorry, I do not understand what possible symptoms you are describing. Try a different description."}
    predicted_disease = predictDiseases(matched_symptoms)
    precautions = get_precautions(predicted_disease[0])

    mess= "After analyzing your symtoms, its appears you may have a " + predicted_disease[0] + ". Precautions you should take are " + precautions+"."
    return {"message": mess}