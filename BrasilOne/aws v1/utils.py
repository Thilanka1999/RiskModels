import pandas as pd
def feature_engineering(df):
    df['score_4_minus_score_3'] = df['score_4'] - df['score_3']
    df['avg_score_5_6'] = (df['score_5'] + df['score_6']) / 2

    # 2. Financial Behavior Features (Handling Zero Division)
    df['debt_to_income'] = df['last_amount_borrowed'] / (df['income'] + 1e-9)  # Add small constant
    df['credit_utilization'] = df['last_amount_borrowed'] / (df['credit_limit'] + 1e-9) # Add small constant
    df['default_rate'] = df['n_defaulted_loans'] / (df['n_accounts'] + 1e-9) # Add small constant
    df['reported_income_div_income'] = df['reported_income'] / (df['income'] + 1e-9) # Add small constant


    # 4. External Data Features
    # df['fraud_score_times_score_1'] = df['external_data_provider_fraud_score'] * df['score_1']

    # 5. Demographic/Location Features (Example - state-level default rate)
    # state_default_rates = df.groupby('state')['target_default'].mean().to_dict()
    # df['state_default_rate'] = df['state'].map(state_default_rates)

    # 6. Facebook Profile Interactions (Example - combining with another feature)
    # df['facebook_profile_times_income'] = df['facebook_profile'] * df['income']

    df['fraud_score_bin'] = pd.cut(df['external_data_provider_fraud_score'], bins=[0, 700, 800, 900, 1000], labels=[0, 1, 2, 3], right = False)
    df['fraud_score_bin'] = df['fraud_score_bin'].astype(int)
    # df['facebook_income_credit'] = df['facebook_profile'] * df['income'] * df['credit_limit']
    df['credit_available'] = df['credit_limit'] - df['last_amount_borrowed'] # Available credit
    df['income_per_account'] = df['income'] / (df['n_accounts'] + 1e-9) # Income per account
    df['loan_amount_to_income'] = df['last_amount_borrowed'] / (df['income'] + 1e-9)
    df['n_accounts_to_credit_limit'] = df['n_accounts'] / (df['credit_limit'] + 1e-9)

    # 8. Interactions between existing engineered features
    df['debt_to_income_x_default_rate'] = df['debt_to_income'] * df['default_rate']
    df['credit_utilization_x_fraud_score'] = df['credit_utilization'] * df['external_data_provider_fraud_score']

    # 9. Polynomial features
    df['income_sq'] = df['income']**2
    df['last_amount_borrowed_sq'] = df['last_amount_borrowed']**2

    return df


def additional_feature_engineering(df):
    # 1. Interaction Features
    df['score_1_x_score_2'] = df['score_1'] * df['score_2']
    df['score_1_x_facebook_profile'] = df['score_1'] * df['facebook_profile']
    df['score_2_x_facebook_profile'] = df['score_2'] * df['facebook_profile']
    df['fraud_score_bin_x_score_1'] = df['fraud_score_bin'].astype(int) * df['score_1']
    df['fraud_score_bin_x_score_2'] = df['fraud_score_bin'].astype(int) * df['score_2']

    # 2. Aggregation Features
    state_avg_score_1 = df.groupby('state')['score_1'].mean().to_dict()
    df['state_avg_score_1'] = df['state'].map(state_avg_score_1)

    state_avg_score_2 = df.groupby('state')['score_2'].mean().to_dict()
    df['state_avg_score_2'] = df['state'].map(state_avg_score_2)

    real_state_avg_facebook_profile = df.groupby('real_state')['facebook_profile'].mean().to_dict()
    df['real_state_avg_facebook_profile'] = df['real_state'].map(real_state_avg_facebook_profile)

    shipping_state_avg_facebook_profile = df.groupby('shipping_state')['facebook_profile'].mean().to_dict()
    df['shipping_state_avg_facebook_profile'] = df['shipping_state'].map(shipping_state_avg_facebook_profile)

    # 3. Ratio Features
    df['score_1_div_score_2'] = df['score_1'] / (df['score_2'] + 1e-9)
    df['facebook_profile_div_score_1'] = df['facebook_profile'] / (df['score_1'] + 1e-9)
    df['facebook_profile_div_score_2'] = df['facebook_profile'] / (df['score_2'] + 1e-9)

    # 4. Difference Features
    df['score_1_minus_score_2'] = df['score_1'] - df['score_2']
    df['facebook_profile_minus_score_1'] = df['facebook_profile'] - df['score_1']
    df['facebook_profile_minus_score_2'] = df['facebook_profile'] - df['score_2']

    # 5. Binning and Encoding Interactions
    df['score_1_bin'] = pd.cut(df['score_1'], bins=[0, 500, 700, 900, 1000], labels=[0, 1, 2, 3], right=False).astype(int)
    df['score_2_bin'] = pd.cut(df['score_2'], bins=[0, 500, 700, 900, 1000], labels=[0, 1, 2, 3], right=False).astype(int)

    df['score_1_bin_x_score_2_bin'] = df['score_1_bin'] * df['score_2_bin']
    df['score_1_bin_x_fraud_score_bin'] = df['score_1_bin'] * df['fraud_score_bin'].astype(int)
    df['score_2_bin_x_fraud_score_bin'] = df['score_2_bin'] * df['fraud_score_bin'].astype(int)

    # 6. State-based Features
    df['state_x_real_state'] = df['state'] + df['real_state']
    df['state_x_shipping_state'] = df['state'] + df['shipping_state']

    state_real_state_avg_score_1 = df.groupby('state_x_real_state')['score_1'].mean().to_dict()
    df['state_real_state_avg_score_1'] = df['state_x_real_state'].map(state_real_state_avg_score_1)

    state_shipping_state_avg_score_2 = df.groupby('state_x_shipping_state')['score_2'].mean().to_dict()
    df['state_shipping_state_avg_score_2'] = df['state_x_shipping_state'].map(state_shipping_state_avg_score_2)

    # 7. Polynomial Features
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


reasoning = {
    '0': "Very High Risk \n This cluster has the highest default rate (0.135) and shows concerning patterns in borrowing behavior. Borrowers in this cluster have taken large loans frequently (last_amount_borrowed: 0.456) and exhibit high credit utilization (0.051). Their risk scores (score_1, score_2) are moderate, but their debt-to-income ratio (0.340) and default rate (0.0004) suggest a high probability of payment issues. This group requires close monitoring and strict risk management.",

    '1': "High Risk \n Borrowers in this cluster have a high default rate (0.238) and show risky borrowing behavior. They have taken moderate-sized loans (last_amount_borrowed: 0.130) and have a relatively high debt-to-income ratio (0.100). Their credit utilization (0.014) is lower than Cluster 0, but their risk scores (score_1, score_2) are slightly lower, indicating higher risk. Their frequent borrowing and moderate risk scores suggest a higher chance of future payment issues.",

    '2': "Low Risk \n This cluster has the lowest default rate (0.113) and exhibits cautious borrowing behavior. Borrowers in this cluster have taken small loans (last_amount_borrowed: 0.061) and have a low debt-to-income ratio (0.049). Their credit utilization (0.006) is minimal, and their risk scores (score_1, score_2) are moderate. Their consistent repayment behavior and low borrowing frequency indicate strong reliability.",

    '3': "Medium Risk \n Borrowers in this cluster have a moderate default rate (0.117) and show mixed borrowing behavior. They have taken very small loans (last_amount_borrowed: 0.008) and have a low debt-to-income ratio (0.007). Their credit utilization (0.001) is minimal, but their risk scores (score_1, score_2) show more variation, suggesting the need for closer monitoring. Their behavior is generally positive but requires caution.",

    '4': "Low-Medium Risk \n This cluster has a low default rate (0.119) and shows good overall reliability. Borrowers in this cluster have taken moderate-sized loans (last_amount_borrowed: 0.064) and have a low debt-to-income ratio (0.051). Their credit utilization (0.007) is minimal, and their risk scores (score_1, score_2) are moderate. Their behavior suggests slightly more caution is needed compared to low-risk borrowers, but they are generally reliable."
}