import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
import joblib
from io import BytesIO, StringIO
import xgboost

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Feature engineering functions
def feature_engineering(df):
    df['score_4_minus_score_3'] = df['score_4'] - df['score_3']
    df['avg_score_5_6'] = (df['score_5'] + df['score_6']) / 2

    # Financial Behavior Features (Handling Zero Division)
    df['debt_to_income'] = df['last_amount_borrowed'] / (df['income'] + 1e-9)
    df['credit_utilization'] = df['last_amount_borrowed'] / (df['credit_limit'] + 1e-9)
    df['default_rate'] = df['n_defaulted_loans'] / (df['n_accounts'] + 1e-9)
    df['reported_income_div_income'] = df['reported_income'] / (df['income'] + 1e-9)

    df['fraud_score_bin'] = pd.cut(df['external_data_provider_fraud_score'], 
                                  bins=[0, 700, 800, 900, 1000], 
                                  labels=[0, 1, 2, 3], 
                                  right=False)
    df['fraud_score_bin'] = df['fraud_score_bin'].astype(int)
    df['credit_available'] = df['credit_limit'] - df['last_amount_borrowed']
    df['income_per_account'] = df['income'] / (df['n_accounts'] + 1e-9)
    df['loan_amount_to_income'] = df['last_amount_borrowed'] / (df['income'] + 1e-9)
    df['n_accounts_to_credit_limit'] = df['n_accounts'] / (df['credit_limit'] + 1e-9)

    # Interactions between existing engineered features
    df['debt_to_income_x_default_rate'] = df['debt_to_income'] * df['default_rate']
    df['credit_utilization_x_fraud_score'] = df['credit_utilization'] * df['external_data_provider_fraud_score']

    # Polynomial features
    df['income_sq'] = df['income']**2
    df['last_amount_borrowed_sq'] = df['last_amount_borrowed']**2

    return df

def additional_feature_engineering(df):
    # Interaction Features
    df['score_1_x_score_2'] = df['score_1'] * df['score_2']
    df['score_1_x_facebook_profile'] = df['score_1'] * df['facebook_profile']
    df['score_2_x_facebook_profile'] = df['score_2'] * df['facebook_profile']
    df['fraud_score_bin_x_score_1'] = df['fraud_score_bin'].astype(int) * df['score_1']
    df['fraud_score_bin_x_score_2'] = df['fraud_score_bin'].astype(int) * df['score_2']

    # Aggregation Features
    state_avg_score_1 = df.groupby('state')['score_1'].mean().to_dict()
    df['state_avg_score_1'] = df['state'].map(state_avg_score_1)

    state_avg_score_2 = df.groupby('state')['score_2'].mean().to_dict()
    df['state_avg_score_2'] = df['state'].map(state_avg_score_2)

    real_state_avg_facebook_profile = df.groupby('real_state')['facebook_profile'].mean().to_dict()
    df['real_state_avg_facebook_profile'] = df['real_state'].map(real_state_avg_facebook_profile)

    shipping_state_avg_facebook_profile = df.groupby('shipping_state')['facebook_profile'].mean().to_dict()
    df['shipping_state_avg_facebook_profile'] = df['shipping_state'].map(shipping_state_avg_facebook_profile)

    # Ratio Features
    df['score_1_div_score_2'] = df['score_1'] / (df['score_2'] + 1e-9)
    df['facebook_profile_div_score_1'] = df['facebook_profile'] / (df['score_1'] + 1e-9)
    df['facebook_profile_div_score_2'] = df['facebook_profile'] / (df['score_2'] + 1e-9)

    # Difference Features
    df['score_1_minus_score_2'] = df['score_1'] - df['score_2']
    df['facebook_profile_minus_score_1'] = df['facebook_profile'] - df['score_1']
    df['facebook_profile_minus_score_2'] = df['facebook_profile'] - df['score_2']

    # Binning and Encoding Interactions
    df['score_1_bin'] = pd.cut(df['score_1'], bins=[0, 500, 700, 900, 1000], labels=[0, 1, 2, 3], right=False).astype(int)
    df['score_2_bin'] = pd.cut(df['score_2'], bins=[0, 500, 700, 900, 1000], labels=[0, 1, 2, 3], right=False).astype(int)

    df['score_1_bin_x_score_2_bin'] = df['score_1_bin'] * df['score_2_bin']
    df['score_1_bin_x_fraud_score_bin'] = df['score_1_bin'] * df['fraud_score_bin'].astype(int)
    df['score_2_bin_x_fraud_score_bin'] = df['score_2_bin'] * df['fraud_score_bin'].astype(int)

    # State-based Features
    df['state_x_real_state'] = df['state'] + df['real_state']
    df['state_x_shipping_state'] = df['state'] + df['shipping_state']

    state_real_state_avg_score_1 = df.groupby('state_x_real_state')['score_1'].mean().to_dict()
    df['state_real_state_avg_score_1'] = df['state_x_real_state'].map(state_real_state_avg_score_1)

    state_shipping_state_avg_score_2 = df.groupby('state_x_shipping_state')['score_2'].mean().to_dict()
    df['state_shipping_state_avg_score_2'] = df['state_x_shipping_state'].map(state_shipping_state_avg_score_2)

    # Polynomial Features
    df['score_1_sq'] = df['score_1']**2
    df['score_2_sq'] = df['score_2']**2
    df['facebook_profile_sq'] = df['facebook_profile']**2

    return df

def inference_validator(user_input):
    required_columns = [
    'score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'last_amount_borrowed',
    'last_borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies',
    'n_defaulted_loans', 'n_accounts', 'n_issues',
    'external_data_provider_credit_checks_last_year', 'external_data_provider_credit_checks_last_month',
    'external_data_provider_email_seen_before', 'reported_income', 'application_time_in_funnel',
    'external_data_provider_fraud_score', 'shipping_state', 'facebook_profile', 'state', 'score_1', 'score_2'
    ]      

    for col in required_columns:
        if col not in user_input:
            user_input[col] = None 
    
    return user_input

# Reasoning dictionary for results interpretation
reasoning_dict = {
    '0': "Very High Risk \n This cluster has the highest default rate (0.135) and shows concerning patterns in borrowing behavior. Borrowers in this cluster have taken large loans frequently (last_amount_borrowed: 0.456) and exhibit high credit utilization (0.051). Their risk scores (score_1, score_2) are moderate, but their debt-to-income ratio (0.340) and default rate (0.0004) suggest a high probability of payment issues. This group requires close monitoring and strict risk management.",
    '1': "High Risk \n Borrowers in this cluster have a high default rate (0.238) and show risky borrowing behavior. They have taken moderate-sized loans (last_amount_borrowed: 0.130) and have a relatively high debt-to-income ratio (0.100). Their credit utilization (0.014) is lower than Cluster 0, but their risk scores (score_1, score_2) are slightly lower, indicating higher risk. Their frequent borrowing and moderate risk scores suggest a higher chance of future payment issues.",
    '2': "Low Risk \n This cluster has the lowest default rate (0.113) and exhibits cautious borrowing behavior. Borrowers in this cluster have taken small loans (last_amount_borrowed: 0.061) and have a low debt-to-income ratio (0.049). Their credit utilization (0.006) is minimal, and their risk scores (score_1, score_2) are moderate. Their consistent repayment behavior and low borrowing frequency indicate strong reliability.",
    '3': "Medium Risk \n Borrowers in this cluster have a moderate default rate (0.117) and show mixed borrowing behavior. They have taken very small loans (last_amount_borrowed: 0.008) and have a low debt-to-income ratio (0.007). Their credit utilization (0.001) is minimal, but their risk scores (score_1, score_2) show more variation, suggesting the need for closer monitoring. Their behavior is generally positive but requires caution.",
    '4': "Low-Medium Risk \n This cluster has a low default rate (0.119) and shows good overall reliability. Borrowers in this cluster have taken moderate-sized loans (last_amount_borrowed: 0.064) and have a low debt-to-income ratio (0.051). Their credit utilization (0.007) is minimal, and their risk scores (score_1, score_2) are moderate. Their behavior suggests slightly more caution is needed compared to low-risk borrowers, but they are generally reliable."
}


def model_fn(model_dir):
    """
    Load model artifacts from the model_dir for SageMaker model serving
    
    Args:
        model_dir (str): Directory where model artifacts are stored
        
    Returns:
        dict: Dictionary containing all loaded model artifacts
    """
    try:
        logger.info(f"Loading model from: {model_dir}")
        
        # Load all artifacts from local model_dir
        artifacts = {}
        
        # Load numerical and categorical imputers (using joblib)
        artifacts['nimputer'] = joblib.load(os.path.join(model_dir, 'nimputer.joblib'))
        logger.info("Loaded nimputer")
        
        artifacts['cimputer'] = joblib.load(os.path.join(model_dir, 'cimputer.joblib'))
        logger.info("Loaded cimputer")
        
        # Load label encoders (still using pickle)
        artifacts['label_encoders'] = joblib.load(os.path.join(model_dir, 'label_encoders.joblib'))
        logger.info("Loaded label_encoders")
        
        # Load preprocessor (using joblib)
        # artifacts['inference_preprocessor'] = joblib.load(os.path.join(model_dir, 'inference_preprocessor.joblib'))
        # logger.info("Loaded inference_preprocessor")
        
        # Load feature list (JSON remains unchanged)
        with open(os.path.join(model_dir, 'before_feature.json'), 'r') as f:
            artifacts['before_columns'] = json.load(f)
        logger.info("Loaded before_columns")

        # Load the model
        model_path = os.path.join(model_dir, 'xgb_model.json')
        model = xgboost.Booster()
        model.load_model(model_path)
        artifacts['xgb'] = model
        logger.info("Loaded xgb model")
        
        # Add the reasoning dictionary
        artifacts['reasoning'] = reasoning_dict
        
        logger.info("All model artifacts loaded successfully")
        return artifacts
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    
    Args:
        request_body: The request body
        request_content_type: The request content type
        
    Returns:
        dict: Input data in dictionary format
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        try:
            input_data = json.loads(request_body)
            logger.info(f"Parsed input data: {str(input_data)[:100]}...")
            return input_data
        except Exception as e:
            logger.error(f"Error parsing JSON input: {str(e)}")
            raise ValueError(f"Error parsing JSON input: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Only application/json is supported.")

def predict_fn(input_data, model_artifacts):
    """
    Apply model to the input data
    
    Args:
        input_data: Input data (from input_fn)
        model_artifacts: Model artifacts (from model_fn)
        
    Returns:
        dict: Prediction result
    """
    expected_columns = [
    'score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'last_amount_borrowed',
    'last_borrowed_in_months', 'credit_limit', 'income', 'ok_since', 'n_bankruptcies',
    'n_defaulted_loans', 'n_accounts', 'n_issues', 'external_data_provider_credit_checks_last_year',
    'external_data_provider_email_seen_before', 'reported_income', 'application_time_in_funnel',
    'external_data_provider_credit_checks_last_month', 'external_data_provider_fraud_score',
    'shipping_state', 'facebook_profile', 'state', 'score_1', 'score_2', 'real_state'
    ]
    pd.set_option("display.max_rows", None)  # Show all rows
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", 1000)  # Increase column width
    pd.set_option("display.max_colwidth", None) 

    try:
        logger.info("Starting prediction process")
        
        if isinstance(input_data, str):
            input_dict = json.loads(input_data)
        else:
            input_dict = input_data
        logger.info(f"Deserialized input data: {input_dict}")
        
        # Convert dictionary to DataFrame
        user_df = pd.DataFrame([input_dict], columns=expected_columns)
        logger.info(f"Initial DataFrame:\n{user_df}")
        
        # Extract model artifacts
        nimputer = model_artifacts['nimputer']
        cimputer = model_artifacts['cimputer']
        label_encoders = model_artifacts['label_encoders']
        # inference_preprocessor = model_artifacts['inference_preprocessor']
        before_columns = model_artifacts['before_columns']
        xgb_model = model_artifacts['xgb']
        reasoning = model_artifacts['reasoning']
        
        # Validate and preprocess the input data
        input_data = inference_validator(user_df)
        logger.info(f"input data:\n{input_data}")
        
        logger.info("Input data validated")
        
        user_df = pd.DataFrame(input_data)
        user_df = user_df.reindex(columns=before_columns, fill_value=None)
        logger.info(f"user_df data:\n{user_df}")
        logger.info(f"Reindexed user_df columns: {user_df.columns.tolist()}")
        logger.info(f"label_encoders: {label_encoders}")
        
        # Apply imputers
        c_features = [f for f in cimputer.feature_names_in_ if f in user_df.columns]
        n_features = [f for f in nimputer.feature_names_in_ if f in user_df.columns]
        logger.info(f"c_features: {c_features}")
        logger.info(f"n_features: {n_features}")
        if c_features:
            user_df[c_features] = cimputer.transform(user_df[c_features])
            logger.info("Applied categorical imputation")
        if n_features:
            user_df[n_features] = nimputer.transform(user_df[n_features])
            logger.info("Applied numerical imputation")
            
        logger.info("Applied imputation")
        logger.info(f"user_df data:\n{user_df}")
        
        # Drop target column if it exists
        if "target_default" in user_df.columns:
            user_df = user_df.drop(columns=["target_default"])
            logger.info("Dropped target_default column")
        
        # Apply label encoding
        for col in c_features:
            if col in user_df.columns and col != "target_default":
                user_df[col] = label_encoders[col].transform(user_df[col])
                logger.info(f"Applied label encoding to column: {col}")
        
        logger.info("Applied label encoding")
        
        # yo_features = [i for i in c_features + n_features if i != "target_default"]
        # user_df = user_df.reindex(columns=yo_features)
        
        # Feature engineering
        user_df = feature_engineering(user_df)
        user_df = additional_feature_engineering(user_df)
        logger.info("Applied feature engineering")
        
        # Transform using the preprocessor
        # # user_processed = inference_preprocessor.transform(user_df)
        # logger.info("Applied feature preprocessing")
        
        # Make prediction
        user_df = xgboost.DMatrix(user_df)
        prediction = xgb_model.predict(user_df)
        # prediction_proba = xgb_model.predict_proba(user_df)
        logger.info(f"Generated prediction: {prediction[0]}")
        
        result = {
            "prediction": np.argmax((prediction[0])).tolist(),
            "reasoning": reasoning[str(np.argmax((prediction[0])))],
            "prediction_proba": prediction[0].tolist()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction_output, accept):
    """
    Serialize the prediction output
    
    Args:
        prediction_output: The prediction output from predict_fn
        accept: The accept content type
        
    Returns:
        The serialized prediction
    """
    logger.info(f"Formatting output with accept type: {accept}")

    logger.info(f"Type of prediction_output: {type(prediction_output)}")
    logger.info(f"Contents of prediction_output: {prediction_output}")
    
    if accept == 'application/json' or accept == '*/*':
        try:
            # def convert_numpy_types(obj):
            #     if isinstance(obj, np.generic):
            #         return obj.item()  # Convert NumPy types to native Python types
            #     elif isinstance(obj, dict):
            #         return {key: convert_numpy_types(value) for key, value in obj.items()}
            #     elif isinstance(obj, (list, tuple)):
            #         return [convert_numpy_types(item) for item in obj]
            #     return obj
            
            # # Apply conversion to the prediction output
            # prediction_output_serializable = convert_numpy_types(prediction_output)
            
            # Serialize to JSON
            json_output = json.dumps(prediction_output)
            logger.info("Successfully serialized prediction to JSON")
            return json_output
        except Exception as e:
            logger.error(f"Error serializing to JSON: {str(e)}")
            raise
    else:
        raise ValueError(f"Unsupported accept type: {accept}. Only application/json is supported.")