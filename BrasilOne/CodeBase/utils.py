import pandas as pd
import numpy as np

def feature_engineering(df):
    temp = pd.DataFrame({'date': pd.to_datetime(df['ApplicationDate'])})
    df['ApplicationDate'] = temp['date'].dt.strftime('%j').astype(int)  

    df['IncomeToDebtRatio'] = df['MonthlyIncome'] / (df['MonthlyDebtPayments'] + 1e-6)
    df['SavingsToIncomeRatio'] = df['SavingsAccountBalance'] / (df['MonthlyIncome'] + 1e-6)
    df['NetWorthToIncomeRatio'] = df['NetWorth'] / (df['MonthlyIncome'] + 1e-6)
    df['HighCreditUtilization'] = (df['CreditCardUtilizationRate'] > 0.7).astype(int)
    # df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['MonthlyIncome'] + 1e-6)
    df['HighDebtToIncome'] = (df['TotalDebtToIncomeRatio'] > 0.3).astype(int)
    # df['CreditAge'] = df['LengthOfCreditHistory'] / 12
    # df['AssetsToLiabilitiesRatio'] = df['TotalAssets'] / (df['TotalLiabilities'] + 1e-6)
    df['JobStability'] = df['JobTenure'] / (df['Age'] - 18 + 1e-6)
    df['PreviousLoanDefaultRate'] = df['PreviousLoanDefaults'] / (df['LengthOfCreditHistory'] + 1e-6)
    # df['PaymentHistoryRatio'] = df['PaymentHistory'] / (df['LengthOfCreditHistory'] + 1e-6)
    df['UtilityBillsPaymentHistoryRatio'] = df['UtilityBillsPaymentHistory'] / (df['LengthOfCreditHistory'] + 1e-6)
    df['InterestRateSpread'] = df['InterestRate'] - df['BaseInterestRate']
    # df['MonthlyLoanPaymentToIncomeRatio'] = df['MonthlyLoanPayment'] / (df['MonthlyIncome'] + 1e-6)
    df['AgeExperienceInteraction'] = df['Age'] * df['Experience']
    # df['CreditScoreDebtToIncomeInteraction'] = df['CreditScore'] * df['DebtToIncomeRatio']
    df['LogMonthlyIncome'] = np.log1p(df['MonthlyIncome'])
    # df['LogLoanAmount'] = np.log1p(df['LoanAmount'])
    df['LogSavingsAccountBalance'] = np.log1p(df['SavingsAccountBalance'])
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['0-30', '30-40', '40-50', '50-60', '60+'])
    df['CreditScoreBin'] = pd.cut(df['CreditScore'], bins=[340, 550, 600, 650, 750], labels=['Poor', 'Fair', 'Good', 'Excellent'])
    # df['CreditScoreSquared'] = df['CreditScore'] ** 2
    # df['DebtToIncomeRatioSquared'] = df['DebtToIncomeRatio'] ** 2
    # df['TotalCreditLinesAndInquiries'] = df['NumberOfOpenCreditLines'] + df['NumberOfCreditInquiries']
    df["RiskScore"] = df["RiskScore"].astype(int)

    df = pd.get_dummies(df, columns=['HighDebtToIncome', 'AgeBin', 'CreditScoreBin', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'EducationLevel', 'LoanPurpose'], drop_first=True)
    # df = df.drop(columns=['AgeBin', 'CreditScoreBin', 'EmploymentStatus', 'MaritalStatus', 'HomeOwnershipStatus', 'EducationLevel', 'LoanPurpose'])
    df = df.drop(columns=['MonthlyIncome', 'SavingsAccountBalance', 'AnnualIncome', 'Age', 'Experience', 'InterestRate', 'NetWorth'])

    return df


def inference_validator(user_input):
    required_columns = ['ApplicationDate', 'Age', 'CreditScore',
       'EmploymentStatus', 'EducationLevel', 'LoanAmount',
       'LoanDuration', 'MaritalStatus', 'NumberOfDependents',
       'HomeOwnershipStatus', 'MonthlyDebtPayments', "Experience",
       'CreditCardUtilizationRate', 'NumberOfOpenCreditLines',
       'NumberOfCreditInquiries', 'DebtToIncomeRatio', 'BankruptcyHistory',
       'LoanPurpose', 'PreviousLoanDefaults', 'PaymentHistory',
       'LengthOfCreditHistory', 'SavingsAccountBalance',
       'CheckingAccountBalance', 'TotalAssets', 'TotalLiabilities',
       'MonthlyIncome', 'UtilityBillsPaymentHistory', 'JobTenure', 'NetWorth',
       'BaseInterestRate', 'InterestRate', 'MonthlyLoanPayment',
       'TotalDebtToIncomeRatio', 'LoanApproved', 'RiskScore'
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