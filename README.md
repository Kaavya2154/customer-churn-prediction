# 📊 Customer Churn Prediction Dashboard

A comprehensive machine learning application built with Python, Pandas, and Streamlit to predict customer churn and visualize key risk indicators.

## 🚀 Features

### 📈 Data Analysis
- **Synthetic Data Generation**: Create realistic customer datasets with configurable sample sizes
- **Interactive Visualizations**: Explore churn patterns through multiple chart types
- **Demographic Analysis**: Understand churn rates by customer segments
- **Risk Factor Identification**: Identify key factors that contribute to customer churn

### 🤖 Machine Learning
- **Random Forest Model**: Advanced ensemble learning for accurate churn prediction
- **Feature Engineering**: Automatic preprocessing and feature scaling
- **Model Performance Metrics**: Comprehensive evaluation with accuracy, confusion matrix, and feature importance
- **Real-time Predictions**: Individual customer churn probability calculation

### 🔍 Interactive Dashboard
- **Multi-tab Interface**: Organized views for different analysis types
- **Customer Segmentation**: Automatic risk-based customer categorization
- **Risk Level Assessment**: High/Medium/Low risk classification
- **Data Filtering**: Advanced filtering options for customer exploration
- **Export Functionality**: Download filtered data and reports

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Access the dashboard**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The dashboard will load automatically

## 📋 Requirements

The following packages are required:

```
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
plotly==5.17.0
seaborn==0.13.0
matplotlib==3.8.2
joblib==1.3.2
```

## 🎯 Usage Guide

### Getting Started

1. **Generate Data**: Use the sidebar to generate synthetic customer data (100-5000 customers)
2. **Train Model**: Click "Train Model" to build the churn prediction model
3. **Explore Dashboard**: Navigate through the different tabs to analyze your data

### Dashboard Tabs

#### 📈 Data Analysis
- **Churn Distribution**: Visualize churn rates by contract type, tenure, and payment method
- **Monthly Charges Analysis**: Compare charges between churned and retained customers
- **Demographic Insights**: Understand churn patterns across customer segments

#### 🎯 Churn Prediction
- **Model Performance**: View accuracy metrics and confusion matrix
- **Feature Importance**: Understand which factors most influence churn
- **Individual Prediction**: Input customer details to get personalized churn probability
- **Risk Assessment**: Visual gauge showing customer risk level

#### 🔍 Risk Indicators
- **Risk Distribution**: Overview of high/medium/low risk customers
- **Risk Factors**: Analysis of common characteristics in high-risk customers
- **Customer Segmentation**: Automatic categorization based on risk and value
- **Targeted Insights**: Identify customers requiring immediate attention

#### 📋 Customer Details
- **Data Table**: Browse and filter customer records
- **Advanced Filtering**: Filter by contract, churn status, tenure, charges, and risk level
- **Export Options**: Download filtered data as CSV files
- **Real-time Updates**: See predictions update as filters change

### Key Features Explained

#### 🎛️ Control Panel (Sidebar)
- **Data Management**: Generate new datasets with different sample sizes
- **Model Training**: Train the machine learning model on your data
- **Real-time Updates**: All changes reflect immediately in the dashboard

#### 📊 Risk Assessment
- **High Risk**: Churn probability > 70%
- **Medium Risk**: Churn probability 40-70%
- **Low Risk**: Churn probability < 40%

#### 🔮 Prediction Form
The individual prediction form includes all relevant customer attributes:
- Demographics (age, senior citizen status, family situation)
- Service details (phone, internet, additional services)
- Contract and billing information
- Financial metrics (monthly and total charges)

## 📊 Data Schema

The synthetic data includes the following customer attributes:

| Field | Type | Description |
|-------|------|-------------|
| customerID | String | Unique customer identifier |
| gender | String | Customer gender |
| SeniorCitizen | Integer | 1 if senior citizen, 0 otherwise |
| Partner | String | Has partner (Yes/No) |
| Dependents | String | Has dependents (Yes/No) |
| tenure | Integer | Months with company |
| PhoneService | String | Has phone service (Yes/No) |
| MultipleLines | String | Multiple phone lines |
| InternetService | String | Type of internet service |
| OnlineSecurity | String | Online security service |
| OnlineBackup | String | Online backup service |
| DeviceProtection | String | Device protection service |
| TechSupport | String | Technical support service |
| StreamingTV | String | Streaming TV service |
| StreamingMovies | String | Streaming movies service |
| Contract | String | Contract type |
| PaperlessBilling | String | Paperless billing (Yes/No) |
| PaymentMethod | String | Payment method |
| MonthlyCharges | Float | Monthly charges in dollars |
| TotalCharges | Float | Total charges in dollars |
| age | Integer | Customer age |
| Churn | Integer | Churn status (1=Yes, 0=No) |

## 🎨 Customization

### Modifying the Model
To use a different machine learning algorithm, edit the `ChurnPredictor` class in `churn_predictor.py`:

```python
# Replace RandomForestClassifier with your preferred model
self.model = YourModelClass()
```

### Adding New Visualizations
Extend the dashboard by adding new charts in the appropriate tab sections of `app.py`.

### Customizing Data Generation
Modify the `generate_synthetic_data` method to create data that better matches your specific use case.

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Package Installation Issues**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

3. **Memory Issues with Large Datasets**
   - Reduce the sample size in the sidebar
   - Close other applications to free up memory

### Performance Tips

- Start with smaller datasets (100-1000 customers) for faster initial testing
- The model training time increases with dataset size
- Use the filtering options to focus on specific customer segments

## 📈 Model Performance

The Random Forest model typically achieves:
- **Accuracy**: 80-85% on synthetic data
- **Feature Importance**: Contract type, tenure, and monthly charges are usually top predictors
- **Prediction Speed**: Real-time predictions for individual customers

