# app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import boto3
from datetime import datetime   
import json


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model from the pickle file
with open('lr_model.pickle', 'rb') as file:
    model = pickle.load(file)

# Define the categorical feature mappings
A1Cresult_categories = ['No', '>7', '>8', 'Norm']
admission_type_desc_categories = ['Urgent', 'Elective', 'Emergency', 'Other']
discharge_category_categories = ['Discharged to Home', 'Transfers to Other Healthcare Facilities','AMA (Against Medical Advice)', 'Other']
admission_category_categories = ['Transfers from Other Facilities', 'Emergency Admission','Physician Referral', 'Other']
race_categories = ['Caucasian', 'AfricanAmerican', 'Other', 'Asian', 'Hispanic']
change_categories = ['Ch', 'No']
diabetesMed_categories= ['Yes', 'No']
payer_code_group_categories= ['Self-Pay/Other', 'Government Programs', 'Private Insurance']
diag_3_cat_categories=['Other', 'Respiratory', 'Diabetes', 'Injury', 'Neoplasms','Circulatory', 'Genitourinary', 'Musculoskeletal', 'Digestive']
 



# Define the list of features expected by the model
FEATURES = [
    'time_in_hospital', 
    'num_lab_procedures',       
    'num_procedures', 
    'num_medications', 
    'number_diagnoses', 
    'metformin',       
    'insulin', 
    'number_outpatient_log', 
    'number_inpatient_log',       
    'number_emergency_log', 
    'Patient_Age', 
    'race_Asian', 
    'race_Caucasian',     
    'race_Hispanic', 
    'race_Other', 
    'A1Cresult_>8', 
    'A1Cresult_No',       
    'A1Cresult_Norm', 
    'change_No', 
    'diabetesMed_Yes',    
    'admission_type_desc_Emergency', 
    'admission_type_desc_Other',       
    'admission_type_desc_Urgent', 
    'discharge_category_Discharged to Home',       
    'discharge_category_Other',       
    'discharge_category_Transfers to Other Healthcare Facilities',       
    'admission_category_Other', 
    'admission_category_Physician Referral',       
    'admission_category_Transfers from Other Facilities',       
    'payer_code_group_Private_Insurance', 
    'payer_code_group_SelfPay_Other',      
    'diag_3_cat_Diabetes', 
    'diag_3_cat_Digestive',      
    'diag_3_cat_Genitourinary', 
    'diag_3_cat_Injury',    
    'diag_3_cat_Musculoskeletal', 
    'diag_3_cat_Neoplasms',       
    'diag_3_cat_Other', 
    'diag_3_cat_Respiratory'
]


 
# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'hospital-readmission-predictions'


def save_response_to_s3(patient_id, input_data, probability):
    # Check for None values before converting
    if patient_id is None:
        raise ValueError("patient_id is None. Cannot convert to integer.")
    if probability is None:
        raise ValueError("probability is None. Cannot convert to integer.")

    # Ensure input_data is JSON serializable
    if isinstance(input_data, np.ndarray):
        input_data = input_data.tolist()

    # Prepare response data
    response_data = {
        "Patient ID": int(patient_id),    
        "Input Request": input_data,
        "prediction": float(probability),
        "timestamp": datetime.now().isoformat()
    }
    
    # Create unique S3 key (file path)
    s3_key = f"responses/{patient_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    
    # Upload to S3
    s3.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=json.dumps(response_data),
        ContentType="application/json"
    )



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
            'number_diagnoses': 0,
            'metformin':0,
            'insulin': 0,
            'number_outpatient_log': 0,
            'number_inpatient_log': 0,
            'number_emergency_log': 0,
            'Patient_Age':0,

             #One-hot encoded race
            'race_Asian':0, 
            'race_Caucasian':0,     
            'race_Hispanic':0,
            'race_Other':0,

            # One-hot encoded A1Cresult
            'A1Cresult_No': 0,
            'A1Cresult_>8': 0,
            'A1Cresult_Norm': 0,

            #One-Hot encoded change
            'change_No':0,
            #One hot encoded diabetesMed
            'diabetesMed_yes':0,

            # One-hot encoded admission_type_desc
            'admission_type_desc_Emergency':0, 
            'admission_type_desc_Other':0, 
            'admission_type_desc_Urgent':0, 
            'discharge_category_Discharged to Home':0,
            'discharge_category_Other':0,     
            'discharge_category_Transfers to Other Healthcare Facilities':0,
            'admission_category_Other':0,
            'admission_category_Physician Referral':0,
            'admission_category_Transfers from Other Facilities':0,

            #One hot encoded payer_code_group
            'payer_code_group_Private_Insurance':0, 
            'payer_code_group_SelfPay_Other':0,


            'diag_3_cat_Diabetes':0, 
            'diag_3_cat_Digestive':0,      
            'diag_3_cat_Genitourinary':0, 
            'diag_3_cat_Injury':0,    
            'diag_3_cat_Musculoskeletal':0, 
            'diag_3_cat_Neoplasms':0,       
            'diag_3_cat_Other':0, 
            'diag_3_cat_Respiratory':0 
        }

        # Validate and process each feature
        patient_id = data.get('Patient_Id')

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

        # 2. Patient Age
        Patient_Age = data.get('Patient_Age')
        if Patient_Age is None:
            return jsonify({'error': 'Patient Age is required.'}), 400
        try:
            Patient_Age = int(Patient_Age)
            if Patient_Age < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'Patient Age must be a non-negative integer and less than 100.'}), 400
        input_data['Patient_Age'] = Patient_Age

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
            return jsonify({'error': 'This field is required.'}), 400
        try:
            number_outpatient_treated = int(number_outpatient_treated)
            if number_outpatient_treated < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'number_outpatient_treated must be a non-negative integer.'}), 400
        # Log transformation
        input_data['number_outpatient_log'] = np.log1p(number_outpatient_treated)

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
        # Log transformation
        input_data['number_inpatient_log'] = np.log1p(number_inpatient_treated)

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
        # Log transformation
        input_data['number_emergency_log'] = np.log1p(number_emergency)


        # 9. race
        race = data.get('race')
        if race is None:
            return jsonify({'error': 'Race is required.'}), 400
        if race not in race_categories:
            return jsonify({'error': f'Invalid value for race. Allowed values are {race_categories}.'}), 400
        # One-hot encoding for race
        for category in race_categories:
            feature_name = f'race_{category}'
            if race != "AfricanAmerican":
                input_data[feature_name] = 1 if race == category else 0
            else:
                input_data[feature_name] = 0


        # 9. change
        change = data.get('change')
        print(change)
        
        if change is None:
            return jsonify({'error': 'Change in Medication is required.'}), 400
        # if change not in change_categories:
        #     return jsonify({'error': f'Invalid value. Allowed values are {change_categories}.'}), 400
        # One-hot encoding for race
        for category in change_categories:
            print(category)
            feature_name = f'change_{category}'
            if change != "Ch":
                input_data[feature_name] = 1 if change == category else 0
            else:
                input_data[feature_name] = 0

        # 9. diag_3_cat
        diag_3_cat = data.get('diag_3_cat')
        if diag_3_cat is None:
            return jsonify({'error': 'Diagnosis category (diag_3_cat) is required.'}), 400
        if diag_3_cat not in diag_3_cat_categories:
            return jsonify({'error': f'Invalid value for diag_3_cat. Allowed values are {diag_3_cat_categories}.'}), 400

        # One-hot encoding for diag_3_cat
        for category in diag_3_cat_categories:
            feature_name = f'diag_3_cat_{category}'
            if category != "Circulatory":
                input_data[feature_name] = 1 if diag_3_cat == category else 0
            else:
                input_data[feature_name] = 0

        # 9. diabetesMed
        diabetesMed = data.get('diabetesMed')
        if diabetesMed is None:
            return jsonify({'error': 'diabetesMed value is required.'}), 400

        if diabetesMed not in diabetesMed_categories:
            return jsonify({'error': f'Invalid value for diabetesMed. Allowed values are {diabetesMed_categories}.'}), 400

        # One-hot encoding for diabetesMed
        for category in diabetesMed_categories:
            feature_name = f'diabetesMed_{category}'
            input_data[feature_name] = 1 if diabetesMed == category else 0

        # 9. Payer Code
        payer_code_group = data.get('payer_code_group')
        if payer_code_group is None:
            return jsonify({'error': 'Payer Code is required.'}), 400
        if payer_code_group not in payer_code_group_categories:
            return jsonify({'error': f'Invalid value for payer code. Allowed values are {payer_code_group_categories}.'}), 400
        # One-hot encoding for race
        for payer in payer_code_group_categories:
            feature_name = f'race_{payer}'
            if payer != "Government Programs":
                input_data[feature_name] = 1 if race == payer else 0
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

        # 11.1 metformin
        metformin = data.get('metformin')
        if metformin is None:
            return jsonify({'error': 'metformin is required.'}), 400
        try:
            metformin = int(metformin)
            if metformin < 0:
                raise ValueError
        except ValueError:
            return jsonify({'error': 'metformin must be a non-negative integer.'}), 400
        input_data['metformin'] = metformin


       
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
        print(input_df)

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

       # After getting the prediction, store the response to S3
        save_response_to_s3(patient_id, input_data, prediction_proba)        

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
