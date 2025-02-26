import pandas as pd
import numpy as np
def feature_engineering(df):
    df['last_amount_borrowed_log'] = np.sqrt(df['last_amount_borrowed'])
    df['last_borrowed_in_months_log'] = np.sqrt(df['last_borrowed_in_months'])
    df['credit_limit_log'] = np.sqrt(df['credit_limit'])
    df['income_log'] = np.sqrt(df['income'])
    df['ok_since_log'] = np.sqrt(df['ok_since'])

    df['n_accounts_log'] = np.sqrt(df['n_accounts'])
    df['n_issues_log'] = np.sqrt(df['n_issues'])
    df['reported_income_log'] = np.sqrt(df['reported_income'])


    df['credit_utilization'] = df['last_amount_borrowed'] / (df['credit_limit'] + 1e-9) # Add small constant
    df['credit_utilization'] = np.sqrt(df['credit_utilization'])

    
    df['reported_income_div_income'] = df['reported_income'] / (df['income'] + 1e-9) # Add small constant
    df['reported_income_div_income'] = np.sqrt(df['reported_income_div_income'])

    df['facebook_profile_times_income'] = df['facebook_profile'] * df['income']
    df['facebook_profile_times_income'] = np.sqrt(df['facebook_profile_times_income'])


    df['credit_available'] = df['credit_limit'] - df['last_amount_borrowed'] # Available credit
    # df['credit_available'] = np.sqrt(df['credit_available'])

    df['income_per_account'] = df['income'] / (df['n_accounts'] + 1e-9) # Income per account
    df['income_per_account'] = np.sqrt(df['income_per_account'])

    df['loan_amount_to_income'] = df['last_amount_borrowed'] / (df['income'] + 1e-9)
    df['loan_amount_to_income'] = np.sqrt(df['loan_amount_to_income'])

    df['n_accounts_to_credit_limit'] = df['n_accounts'] / (df['credit_limit'] + 1e-9)
    df['n_accounts_to_credit_limit'] = np.sqrt(df['n_accounts_to_credit_limit'])


    # 8. Interactions between existing engineered features
    # df['debt_to_income_x_default_rate'] = df['debt_to_income'] * df['default_rate']
    df['credit_utilization_x_fraud_score'] = df['credit_utilization'] * df['external_data_provider_fraud_score']


    return df


def additional_feature_engineering(df):

    real_state_avg_facebook_profile = df.groupby('real_state')['facebook_profile'].mean().to_dict()
    df['real_state_avg_facebook_profile'] = df['real_state'].map(real_state_avg_facebook_profile)

    shipping_state_avg_facebook_profile = df.groupby('shipping_state')['facebook_profile'].mean().to_dict()
    df['shipping_state_avg_facebook_profile'] = df['shipping_state'].map(shipping_state_avg_facebook_profile)


    # 6. State-based Features
    df['state_x_real_state'] = df['state'].astype(int) * df['real_state'].astype(int)
    df['state_x_shipping_state'] = df['state'].astype(int) * df['shipping_state'].astype(int)

    state_real_state_avg_score_1 = df.groupby('state_x_real_state')['score_1'].mean().to_dict()
    df['state_real_state_avg_score_1'] = df['state_x_real_state'].map(state_real_state_avg_score_1)

    # state_shipping_state_avg_score_2 = df.groupby('state_x_shipping_state')['score_2'].mean().to_dict()
    # df['state_shipping_state_avg_score_2'] = df['state_x_shipping_state'].map(state_shipping_state_avg_score_2)

    return df


def inference_validator(user_input):
    required_columns = [
    'score_3', 'score_4', 'score_5', 'score_6', 'risk_rate', 'last_amount_borrowed',
    'last_borrowed_in_months', 'credit_limit', 'income', 'ok_since',
    'n_accounts', 'n_issues',
    'external_data_provider_credit_checks_last_month',
    'facebook_profile',
    'external_data_provider_credit_checks_last_year',
    'external_data_provider_email_seen_before', 'reported_income', 'application_time_in_funnel',
    'external_data_provider_fraud_score', 'shipping_state', 'state', 'score_1'
    ]      

    for col in required_columns:
        if col not in user_input:
            user_input[col] = None 
    
    return user_input


reasoning = {
    '0': "Very High Risk \n This cluster has the highest default rate (0.135) and shows concerning patterns in borrowing behavior. Borrowers in this cluster have taken large loans frequently (last_amount_borrowed: 0.456) and exhibit high credit utilization (0.051). Their risk scores (score_1, score_2) are moderate, but their debt-to-income ratio (0.340) and default rate (0.0004) suggest a high probability of payment issues. This group requires close monitoring and strict risk management.",

    '1': "High Risk \n Borrowers in this cluster have a high default rate (0.238) and show risky borrowing behavior. They have taken moderate-sized loans (last_amount_borrowed: 0.130) and have a relatively high debt-to-income ratio (0.100). Their credit utilization (0.014) is lower than Cluster 0, but their risk scores (score_1, score_2) are slightly lower, indicating higher risk. Their frequent borrowing and moderate risk scores suggest a higher chance of future payment issues.",

    '2': "Low Risk \n This cluster has the lowest default rate (0.113) and exhibits cautious borrowing behavior. Borrowers in this cluster have taken small loans (last_amount_borrowed: 0.061) and have a low debt-to-income ratio (0.049). Their credit utilization (0.006) is minimal, and their risk scores (score_1, score_2) are moderate. Their consistent repayment behavior and low borrowing frequency indicate strong reliability.",

    '3': "Medium Risk \n Borrowers in this cluster have a moderate default rate (0.117) and show mixed borrowing behavior. They have taken very small loans (last_amount_borrowed: 0.008) and have a low debt-to-income ratio (0.007). Their credit utilization (0.001) is minimal, but their risk scores (score_1, score_2) show more variation, suggesting the need for closer monitoring. Their behavior is generally positive but requires caution.",

    '4': "Low-Medium Risk \n This cluster has a low default rate (0.119) and shows good overall reliability. Borrowers in this cluster have taken moderate-sized loans (last_amount_borrowed: 0.064) and have a low debt-to-income ratio (0.051). Their credit utilization (0.007) is minimal, and their risk scores (score_1, score_2) are moderate. Their behavior suggests slightly more caution is needed compared to low-risk borrowers, but they are generally reliable."
}