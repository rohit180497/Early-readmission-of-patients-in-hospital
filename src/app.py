# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model from the pickle file
with open('rf_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Define the categorical feature mappings
max_glu_serum_categories = ['No', '>300', 'Norm', '>200']
A1Cresult_categories = ['No', '>7', '>8', 'Norm']
admission_type_desc_categories = ['Urgent', 'Elective', 'Emergency', 'Other']
discharge_category_categories = ['Discharged to Home', 'Transfers to Other Healthcare Facilities','AMA (Against Medical Advice)', 'Other']
admission_category_categories = ['Transfers from Other Facilities', 'Emergency Admission','Physician Referral', 'Other']




# Define the list of features expected by the model
# Assuming 15 original + 8 one-hot encoded for max_glu_serum and A1Cresult + 2 label encoded
FEATURES = [
    'time_in_hospital', 
    'num_lab_procedures', 
    'num_procedures', 
    'num_medications', 
    'number_emergency',
    'number_diagnoses', 
    'insulin', 
    'diabetesMed', 
    'number_outpatient_treated', 
    'number_inpatient_treated',  
    'max_glu_serum_>300', 
    'max_glu_serum_No', 
    'max_glu_serum_Norm',   
    'A1Cresult_>8', 
    'A1Cresult_No', 
    'A1Cresult_Norm',
    'admission_type_desc_Emergency', 
    'admission_type_desc_Other', 
    'admission_type_desc_Urgent', 
    'discharge_category_Discharged to Home', 
    'discharge_category_Other',       
    'discharge_category_Transfers to Other Healthcare Facilities', 
    'admission_category_Other', 
    'admission_category_Physician Referral',  
    'admission_category_Transfers from Other Facilities'
]


 

@app.route('/')
def home():
    return render_template('index.html')  # Serve the frontend HTML

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the POST request
        data = request.get_json(force=True)

        # Initialize a dictionary for all features
        input_data = {
            'time_in_hospital': 0,
            'num_lab_procedures': 0,
            'num_procedures': 0,
            'num_medications': 0,
            'number_emergency': 0,
            'number_diagnoses': 0,
            'number_outpatient_treated': 0,
            'number_inpatient_treated': 0,
            'insulin': 0,
            'diabetesMed': 0,
            'admission_type_desc': 0,
            'discharge_category': 0,
            'admission_category': 0,
            # One-hot encoded max_glu_serum
            'max_glu_serum_No': 0,
            'max_glu_serum_>300': 0,
            'max_glu_serum_Norm': 0,
            # One-hot encoded A1Cresult
            'A1Cresult_No': 0,
            'A1Cresult_>8': 0,
            'A1Cresult_Norm': 0,
            # One-hot encoded admission_type_desc
            'admission_type_desc_Emergency':0, 
            'admission_type_desc_Other':0, 
            'admission_type_desc_Urgent':0, 
            'discharge_category_Discharged to Home':0,
            'discharge_category_Other':0,     
            'discharge_category_Transfers to Other Healthcare Facilities':0,
            'admission_category_Other':0,
            'admission_category_Physician Referral':0,
            'admission_category_Transfers from Other Facilities':0
        }

        # Validate and process each feature

        # 1. time_in_hospital
        time_in_hospital = data.get('time_in_hospital')
        if time_in_hospital is None:
            return jsonify({'error': 'time_in_hospital is required.'}), 400
        try:
            time_in_hospital = int(time_in_hospital)
            if time_in_hospital < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'time_in_hospital must be a non-negative integer.'}), 400
        input_data['time_in_hospital'] = time_in_hospital

        # 2. num_lab_procedures
        num_lab_procedures = data.get('num_lab_procedures')
        if num_lab_procedures is None:
            return jsonify({'error': 'num_lab_procedures is required.'}), 400
        try:
            num_lab_procedures = int(num_lab_procedures)
            if num_lab_procedures < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'num_lab_procedures must be a non-negative integer.'}), 400
        input_data['num_lab_procedures'] = num_lab_procedures

        # 3. num_procedures
        num_procedures = data.get('num_procedures')
        if num_procedures is None:
            return jsonify({'error': 'num_procedures is required.'}), 400
        try:
            num_procedures = int(num_procedures)
            if num_procedures < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'num_procedures must be a non-negative integer.'}), 400
        input_data['num_procedures'] = num_procedures

        # 4. num_medications
        num_medications = data.get('num_medications')
        if num_medications is None:
            return jsonify({'error': 'num_medications is required.'}), 400
        try:
            num_medications = int(num_medications)
            if num_medications < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'num_medications must be a non-negative integer.'}), 400
        input_data['num_medications'] = num_medications

        # 5. number_emergency
        number_emergency = data.get('number_emergency')
        if number_emergency is None:
            return jsonify({'error': 'number_emergency is required.'}), 400
        try:
            number_emergency = int(number_emergency)
            if number_emergency < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'number_emergency must be a non-negative integer.'}), 400
        input_data['number_emergency'] = number_emergency

        # 6. number_diagnoses
        number_diagnoses = data.get('number_diagnoses')
        if number_diagnoses is None:
            return jsonify({'error': 'number_diagnoses is required.'}), 400
        try:
            number_diagnoses = int(number_diagnoses)
            if number_diagnoses < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'number_diagnoses must be a non-negative integer.'}), 400
        input_data['number_diagnoses'] = number_diagnoses

        # 7. number_outpatient_treated
        number_outpatient_treated = data.get('number_outpatient_treated')
        if number_outpatient_treated is None:
            return jsonify({'error': 'number_outpatient_treated is required.'}), 400
        try:
            number_outpatient_treated = int(number_outpatient_treated)
            if number_outpatient_treated < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'number_outpatient_treated must be a non-negative integer.'}), 400
        input_data['number_outpatient_treated'] = number_outpatient_treated

        # 8. number_inpatient_treated
        number_inpatient_treated = data.get('number_inpatient_treated')
        if number_inpatient_treated is None:
            return jsonify({'error': 'number_inpatient_treated is required.'}), 400
        try:
            number_inpatient_treated = int(number_inpatient_treated)
            if number_inpatient_treated < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'number_inpatient_treated must be a non-negative integer.'}), 400
        input_data['number_inpatient_treated'] = number_inpatient_treated

        # 9. max_glu_serum
        max_glu_serum = data.get('max_glu_serum')
        if max_glu_serum is None:
            return jsonify({'error': 'max_glu_serum is required.'}), 400
        if max_glu_serum not in max_glu_serum_categories:
            return jsonify({'error': f'Invalid value for max_glu_serum. Allowed values are {max_glu_serum_categories}.'}), 400
        # One-hot encoding for max_glu_serum
        for category in max_glu_serum_categories:
            feature_name = f'max_glu_serum_{category}'    
            if max_glu_serum != ">200":
                input_data[feature_name] = 1 if max_glu_serum == category else 0
            else:
                input_data[feature_name] = 0


        # 10. A1Cresult
        A1Cresult = data.get('A1Cresult')
        if A1Cresult is None:
            return jsonify({'error': 'A1Cresult is required.'}), 400
        if A1Cresult not in A1Cresult_categories:
            return jsonify({'error': f'Invalid value for A1Cresult. Allowed values are {A1Cresult_categories}.'}), 400
        # One-hot encoding for A1Cresult
        for category in A1Cresult_categories:
            feature_name = f'A1Cresult_{category}'
            if A1Cresult != ">7":
                input_data[feature_name] = 1 if A1Cresult == category else 0
            else:
                input_data[feature_name] = 0

        # 11. insulin
        insulin = data.get('insulin')
        if insulin is None:
            return jsonify({'error': 'insulin is required.'}), 400
        try:
            insulin = int(insulin)
            if insulin < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'insulin must be a non-negative integer.'}), 400
        input_data['insulin'] = insulin

        # 12. diabetesMed
        diabetesMed = data.get('diabetesMed')
        if diabetesMed is None:
            return jsonify({'error': 'diabetesMed is required.'}), 400
        if isinstance(diabetesMed, bool):
            diabetesMed = int(diabetesMed)
        elif diabetesMed in ['true', 'True', '1']:
            diabetesMed = 1
        elif diabetesMed in ['false', 'False', '0']:
            diabetesMed = 0
        else:
            return jsonify({'error': 'diabetesMed must be a boolean value.'}), 400
        input_data['diabetesMed'] = diabetesMed

        # 13. admission_type_desc
        admission_type_desc = data.get('admission_type_desc')
        if admission_type_desc is None:
            return jsonify({'error': 'admission_type_desc is required.'}), 400
        if admission_type_desc not in admission_type_desc_categories:
            return jsonify({'error': f'Invalid value for admission_type_desc. Allowed values are {admission_type_desc_categories}.'}), 400
        # One-hot encoding for admission_type_desc
        for category in admission_type_desc_categories:
            feature_name = f'admission_type_desc_{category}'
            if admission_type_desc != "Elective":
                input_data[feature_name] = 1 if admission_type_desc == category else 0
            else:
                input_data[feature_name] = 0

        # 14. discharge_category
        discharge_category = data.get('discharge_category')
        if discharge_category is None:
            return jsonify({'error': 'discharge_category is required.'}), 400
        if discharge_category not in discharge_category_categories:
            return jsonify({'error': f'Invalid value for discharge_category. Allowed values are {discharge_category_categories}.'}), 400
        # One-hot encoding for discharge_category
        for category in discharge_category_categories:
            feature_name = f'discharge_category_{category}'
            if discharge_category != "AMA (Against Medical Advice)":
                input_data[feature_name] = 1 if discharge_category == category else 0
            else:
                input_data[feature_name] = 0

        # 15. admission_category
        admission_category = data.get('admission_category')
        admission_category = data.get('admission_category')
        if admission_category is None:
            return jsonify({'error': 'admission_category is required.'}), 400
        if admission_category not in admission_category_categories:
            return jsonify({'error': f'Invalid value for admission_category. Allowed values are {admission_category_categories}.'}), 400
        # One-hot encoding for admission_category
        for category in admission_category_categories:
            feature_name = f'admission_category_{category}'
            if admission_category != "Emergency Admission":
                input_data[feature_name] = 1 if admission_category == category else 0
            else:
                input_data[feature_name] = 0

        # Create DataFrame in the order of FEATURES
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Ensure all required features are present
        if input_df.shape[1] != len(FEATURES):
            return jsonify({'error': 'Incorrect number of features provided.'}), 400

        # Convert the DataFrame to a numpy array
        input_array = input_df.values

        # Make prediction
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0][1]  # Probability of class '1'

        #Prepare the response
        response = {
            'prediction': int(prediction),
            'probability': float(prediction_proba)
        }

       
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
