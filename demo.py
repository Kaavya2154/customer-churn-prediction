#!/usr/bin/env python3
"""
Demo script for Customer Churn Prediction
This script demonstrates the core functionality without the Streamlit interface.
"""

from churn_predictor import ChurnPredictor
import pandas as pd
import numpy as np

def main():
    print("🚀 Customer Churn Prediction Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Generate sample data
    print("\n📊 Generating synthetic customer data...")
    data = predictor.generate_synthetic_data(n_samples=500)
    print(f"✅ Generated {len(data)} customer records")
    
    # Show data overview
    print(f"\n📈 Data Overview:")
    print(f"   - Total customers: {len(data)}")
    print(f"   - Churn rate: {data['Churn'].mean()*100:.1f}%")
    print(f"   - Average tenure: {data['tenure'].mean():.1f} months")
    print(f"   - Average monthly charges: ${data['MonthlyCharges'].mean():.2f}")
    
    # Train model
    print("\n🤖 Training churn prediction model...")
    metrics = predictor.train_model(data)
    print(f"✅ Model trained successfully!")
    print(f"   - Accuracy: {metrics['accuracy']:.3f}")
    
    # Show feature importance
    print(f"\n🔍 Top 5 Most Important Features:")
    feature_importance = metrics['feature_importance']
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        print(f"   {i}. {feature}: {importance:.3f}")
    
    # Demo individual prediction
    print(f"\n🔮 Demo: Predicting churn for a sample customer...")
    
    # Create a sample customer (high-risk profile)
    sample_customer = pd.DataFrame({
        'customerID': ['DEMO_CUSTOMER'],
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [6],  # Short tenure
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],  # No security
        'OnlineBackup': ['No'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['Yes'],
        'Contract': ['Month-to-month'],  # Month-to-month contract
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],  # Electronic check
        'MonthlyCharges': [95.0],  # High charges
        'TotalCharges': [570.0],
        'age': [35],
        'Churn': [0]  # Dummy value
    })
    
    # Make prediction
    prediction = predictor.predict_churn(sample_customer)
    
    print(f"   Customer Profile:")
    print(f"   - Contract: Month-to-month")
    print(f"   - Tenure: 6 months")
    print(f"   - Monthly Charges: $95.00")
    print(f"   - Payment Method: Electronic check")
    print(f"   - Online Security: No")
    
    print(f"\n   Prediction Results:")
    print(f"   - Churn Probability: {prediction['churn_probability']:.3f}")
    print(f"   - Risk Level: {prediction['risk_level']}")
    print(f"   - Predicted Outcome: {'Will Churn' if prediction['churn_prediction'] == 1 else 'Will Not Churn'}")
    
    # Analyze risk factors
    print(f"\n⚠️  Risk Analysis:")
    if prediction['churn_probability'] > 0.7:
        print("   🔴 HIGH RISK - Immediate attention required!")
        print("   📋 Recommended actions:")
        print("      - Offer retention incentives")
        print("      - Assign dedicated account manager")
        print("      - Review service quality")
    elif prediction['churn_probability'] > 0.4:
        print("   🟡 MEDIUM RISK - Monitor closely")
        print("   📋 Recommended actions:")
        print("      - Regular check-ins")
        print("      - Proactive support")
        print("      - Service optimization")
    else:
        print("   🟢 LOW RISK - Customer appears stable")
        print("   📋 Recommended actions:")
        print("      - Maintain current service level")
        print("      - Upsell opportunities")
        print("      - Referral programs")
    
    print(f"\n🎯 To explore the full interactive dashboard:")
    print(f"   Run: streamlit run app.py")
    print(f"   Then open: http://localhost:8501")
    
    print(f"\n✨ Demo completed successfully!")

if __name__ == "__main__":
    main()
