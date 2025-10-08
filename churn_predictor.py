import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic customer data for demonstration"""
        np.random.seed(42)
        
        # Customer demographics
        age = np.random.normal(40, 15, n_samples).astype(int)
        age = np.clip(age, 18, 80)
        
        # Account features
        tenure_months = np.random.exponential(24, n_samples).astype(int)
        tenure_months = np.clip(tenure_months, 1, 72)
        
        monthly_charges = np.random.normal(65, 20, n_samples)
        monthly_charges = np.clip(monthly_charges, 20, 120)
        
        total_charges = tenure_months * monthly_charges + np.random.normal(0, 100, n_samples)
        
        # Service features
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
        phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
        multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.3, 0.3, 0.4])
        
        # Contract and billing
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2])
        paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
        payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                        n_samples, p=[0.3, 0.2, 0.25, 0.25])
        
        # Additional services
        online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.4, 0.3])
        
        # Create DataFrame
        data = {
            'customerID': [f'CUST_{i:06d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': tenure_months,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'age': age
        }
        
        df = pd.DataFrame(data)
        
        # Generate churn based on realistic patterns
        churn_prob = np.zeros(n_samples)
        
        # Higher churn for month-to-month contracts
        churn_prob += np.where(df['Contract'] == 'Month-to-month', 0.3, 0)
        churn_prob += np.where(df['Contract'] == 'One year', 0.1, 0)
        churn_prob += np.where(df['Contract'] == 'Two year', 0.05, 0)
        
        # Higher churn for higher monthly charges
        churn_prob += (df['MonthlyCharges'] - df['MonthlyCharges'].mean()) / df['MonthlyCharges'].std() * 0.1
        
        # Higher churn for shorter tenure
        churn_prob += (df['tenure'].mean() - df['tenure']) / df['tenure'].std() * 0.15
        
        # Higher churn for electronic check payment
        churn_prob += np.where(df['PaymentMethod'] == 'Electronic check', 0.2, 0)
        
        # Higher churn for no online security
        churn_prob += np.where(df['OnlineSecurity'] == 'No', 0.1, 0)
        
        # Add some randomness
        churn_prob += np.random.normal(0, 0.1, n_samples)
        
        # Convert to binary churn
        df['Churn'] = (churn_prob > np.median(churn_prob)).astype(int)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for machine learning"""
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Convert categorical variables to numeric
        categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                             'MultipleLines', 'InternetService', 'OnlineSecurity', 
                             'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                             'StreamingTV', 'StreamingMovies', 'Contract', 
                             'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_columns:
            df_processed[col] = pd.Categorical(df_processed[col]).codes
        
        # Handle missing values in TotalCharges
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median())
        
        return df_processed
    
    def train_model(self, df):
        """Train the churn prediction model"""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Select features for training
        feature_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                          'PhoneService', 'MultipleLines', 'InternetService', 
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 
                          'Contract', 'PaperlessBilling', 'PaymentMethod', 
                          'MonthlyCharges', 'TotalCharges', 'age']
        
        X = df_processed[feature_columns]
        y = df_processed['Churn']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(feature_columns, self.model.feature_importances_))
        }
    
    def predict_churn(self, customer_data):
        """Predict churn for a single customer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the customer data
        df_processed = self.preprocess_data(customer_data)
        
        # Select features
        feature_columns = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                          'PhoneService', 'MultipleLines', 'InternetService', 
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 
                          'Contract', 'PaperlessBilling', 'PaymentMethod', 
                          'MonthlyCharges', 'TotalCharges', 'age']
        
        X = df_processed[feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        churn_probability = self.model.predict_proba(X_scaled)[0][1]
        churn_prediction = self.model.predict(X_scaled)[0]
        
        return {
            'churn_probability': churn_probability,
            'churn_prediction': churn_prediction,
            'risk_level': 'High' if churn_probability > 0.7 else 'Medium' if churn_probability > 0.4 else 'Low'
        }
    
    def save_model(self, filepath='churn_model.joblib'):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath='churn_model.joblib'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
        else:
            raise FileNotFoundError(f"Model file {filepath} not found")
