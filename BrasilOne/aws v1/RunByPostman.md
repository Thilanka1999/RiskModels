# Assessing Credit Risk Using SageMaker Endpoint 🚀

## Overview

This guide provides instructions on how to send a POST request to the credit risk assessment model's Amazon SageMaker endpoint using Postman. 

## Steps

1. **Create a New Request**

   - Method: `POST`
   - URL: `https://runtime.sagemaker.ap-south-1.amazonaws.com/endpoints/customer-creditrisk-V1/invocations`

2. **Set Headers**

   - `Content-Type: application/json`
   - `Accept: application/json`

3. **Add JSON Body**

   ```json
   {
    "score_3": 800,                     // Min: 0, Max: 990
    "score_4": null,                    // Min: 86.191572, Max: 113.978234
    "score_5": 0.85,                    // Min: 0.000035, Max: 0.999973
    "score_6": 98,                      // Min: 60.663039, Max: 142.192400
    "risk_rate": 0.10,                  // Min: 0.000000, Max: 0.900000
    "last_amount_borrowed": 200,        // Min: 0.000000, Max: 35059.600000
    "last_borrowed_in_months": 36,      // Min: 0.000000, Max: 60.000000
    "credit_limit": 10,                 // Min: 0.000000, Max: 448269.000000
    "income": 120000,                   // Min: 4821.180000, Max: 5000028.000000
    "ok_since": 60,                     // Min: 0.000000, Max: 141.000000
    "n_bankruptcies": 0.0,              // Min: 0.000000, Max: 5.000000
    "n_defaulted_loans": 0,             // Min: 0.000000, Max: 5.000000
    "n_accounts": 10.0,                 // Min: 0.000000, Max: 49.000000
    "n_issues": 1.0,                    // Min: 0.000000, Max: 49.000000
    "external_data_provider_credit_checks_last_year": 1.0, // Min: 0.000000, Max: 1.000000
    "external_data_provider_email_seen_before": 10,        // Min: -1.000000, Max: 59.000000
    "reported_income": 120000,          // Min: 403.000000, Max: 6355500000000000.000000
    "application_time_in_funnel": 300,  // Min: 0.000000, Max: 500.000000
    "external_data_provider_credit_checks_last_month": 0.0, // Min: 0.000000, Max: 3.000000
    "external_data_provider_fraud_score": 100, // Min: 0.000000, Max: 1000.000000
    "shipping_state": null,             // No min/max provided (categorical)
    "facebook_profile": false,          // No min/max provided (boolean)
    "state": null,                      // No min/max provided (categorical)
    "score_1": "1Rk8w4Ucd5yR3KcqZzLdow==", // No min/max provided (encoded string)
    "score_2": null,                    // No min/max provided (null)
    "real_state": null                  // No min/max provided (null)
   }
   ```
 - some info about categorical features.
 - shipping_states => ['BR-MT' 'BR-RS' 'BR-RR' 'BR-RN' 'BR-SP' 'BR-AC' 'BR-MS' 'BR-PE' 'BR-AM'
   'BR-CE' 'BR-SE' 'BR-AP' 'BR-MA' 'BR-BA' 'BR-TO' 'BR-RO' 'BR-SC' 'BR-GO'
   'BR-PR' 'BR-MG' 'BR-ES' 'BR-DF' 'BR-PA' 'BR-PB' 'BR-AL']

 - real_state => 5 different hashed values
 - state => 25 different hashed values
 - score_1 => 7 different hashed values
 - score_2 => 35 different hashed values

 - The input feature set may contain missing keys or null values for certain features.

4. **Configure Authorization** 🔑

   - Type: `AWS Signature`
   - Provide AWS Access Key, Secret Key, Region, and Service Name (`sagemaker`).

## LIMITATIONS

- The current model's prediction heavily depends on whether the user has a Facebook profile.

This guide helps you quickly set up and test your SageMaker endpoint for credit risk assessment using Postman. ✅