import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from churn_predictor import ChurnPredictor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff8800;
        font-weight: bold;
    }
    .risk-low {
        color: #00aa00;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = ChurnPredictor()
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Data generation
        st.subheader("📈 Data Management")
        n_samples = st.slider("Number of customers", 100, 5000, 1000)
        
        if st.button("🔄 Generate New Data", type="primary"):
            with st.spinner("Generating synthetic customer data..."):
                st.session_state.data = st.session_state.predictor.generate_synthetic_data(n_samples)
                st.session_state.model_trained = False
            st.success(f"Generated {n_samples} customer records!")
        
        # Model training
        st.subheader("🤖 Model Training")
        if st.session_state.data is not None:
            if st.button("🎯 Train Model", type="primary"):
                with st.spinner("Training churn prediction model..."):
                    metrics = st.session_state.predictor.train_model(st.session_state.data)
                    st.session_state.model_trained = True
                    st.session_state.metrics = metrics
                st.success("Model trained successfully!")
        else:
            st.info("Generate data first to train the model")
    
    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Overview metrics
        st.subheader("📊 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(data))
        
        with col2:
            churn_rate = data['Churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        with col3:
            avg_tenure = data['tenure'].mean()
            st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
        
        with col4:
            avg_monthly_charges = data['MonthlyCharges'].mean()
            st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Analysis", "🎯 Churn Prediction", "🔍 Risk Indicators", "📋 Customer Details"])
        
        with tab1:
            st.subheader("Data Analysis & Visualizations")
            
            # Churn distribution
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn by contract type
                contract_churn = data.groupby('Contract')['Churn'].agg(['count', 'sum', 'mean']).reset_index()
                contract_churn['churn_rate'] = contract_churn['mean'] * 100
                
                fig_contract = px.bar(
                    contract_churn, 
                    x='Contract', 
                    y='churn_rate',
                    title="Churn Rate by Contract Type",
                    color='churn_rate',
                    color_continuous_scale='Reds'
                )
                fig_contract.update_layout(yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig_contract, use_container_width=True)
            
            with col2:
                # Churn by tenure
                tenure_bins = pd.cut(data['tenure'], bins=6, labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60+'])
                tenure_churn = data.groupby(tenure_bins)['Churn'].mean() * 100
                
                fig_tenure = px.bar(
                    x=tenure_churn.index,
                    y=tenure_churn.values,
                    title="Churn Rate by Tenure",
                    color=tenure_churn.values,
                    color_continuous_scale='Reds'
                )
                fig_tenure.update_layout(xaxis_title="Tenure (months)", yaxis_title="Churn Rate (%)")
                st.plotly_chart(fig_tenure, use_container_width=True)
            
            # Monthly charges vs churn
            col1, col2 = st.columns(2)
            
            with col1:
                fig_charges = px.box(
                    data, 
                    x='Churn', 
                    y='MonthlyCharges',
                    title="Monthly Charges Distribution by Churn Status",
                    color='Churn',
                    color_discrete_map={0: 'blue', 1: 'red'}
                )
                fig_charges.update_layout(xaxis_title="Churn (0=No, 1=Yes)", yaxis_title="Monthly Charges ($)")
                st.plotly_chart(fig_charges, use_container_width=True)
            
            with col2:
                # Payment method churn
                payment_churn = data.groupby('PaymentMethod')['Churn'].mean() * 100
                
                fig_payment = px.pie(
                    values=payment_churn.values,
                    names=payment_churn.index,
                    title="Churn Rate by Payment Method"
                )
                st.plotly_chart(fig_payment, use_container_width=True)
        
        with tab2:
            st.subheader("Churn Prediction")
            
            if st.session_state.model_trained:
                # Model performance metrics
                metrics = st.session_state.metrics
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Model Accuracy", f"{metrics['accuracy']:.3f}")
                    
                    # Confusion matrix
                    cm = metrics['confusion_matrix']
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        color_continuous_scale='Blues'
                    )
                    fig_cm.update_layout(
                        xaxis_title="Predicted",
                        yaxis_title="Actual"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with col2:
                    # Feature importance
                    feature_importance = metrics['feature_importance']
                    importance_df = pd.DataFrame(
                        list(feature_importance.items()),
                        columns=['Feature', 'Importance']
                    ).sort_values('Importance', ascending=True)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance",
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # Individual customer prediction
                st.subheader("🔮 Predict Individual Customer Churn")
                
                # Create input form
                with st.form("customer_prediction"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                        partner = st.selectbox("Partner", ["Yes", "No"])
                        dependents = st.selectbox("Dependents", ["Yes", "No"])
                        tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=24)
                        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
                    
                    with col2:
                        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
                        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                    
                    with col3:
                        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
                        monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=65.0)
                    
                    with col5:
                        total_charges = st.number_input("Total Charges ($)", min_value=100.0, max_value=8000.0, value=1500.0)
                        age = st.number_input("Age", min_value=18, max_value=80, value=40)
                    
                    submitted = st.form_submit_button("🔮 Predict Churn", type="primary")
                    
                    if submitted:
                        # Create customer data
                        customer_data = pd.DataFrame({
                            'customerID': ['CUST_PREDICT'],
                            'gender': ['Male'],
                            'SeniorCitizen': [senior_citizen],
                            'Partner': [partner],
                            'Dependents': [dependents],
                            'tenure': [tenure],
                            'PhoneService': [phone_service],
                            'MultipleLines': [multiple_lines],
                            'InternetService': [internet_service],
                            'OnlineSecurity': [online_security],
                            'OnlineBackup': [online_backup],
                            'DeviceProtection': [device_protection],
                            'TechSupport': [tech_support],
                            'StreamingTV': [streaming_tv],
                            'StreamingMovies': [streaming_movies],
                            'Contract': [contract],
                            'PaperlessBilling': [paperless_billing],
                            'PaymentMethod': [payment_method],
                            'MonthlyCharges': [monthly_charges],
                            'TotalCharges': [total_charges],
                            'age': [age],
                            'Churn': [0]  # Dummy value
                        })
                        
                        # Make prediction
                        prediction = st.session_state.predictor.predict_churn(customer_data)
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Churn Probability", f"{prediction['churn_probability']:.3f}")
                        
                        with col2:
                            prediction_text = "Will Churn" if prediction['churn_prediction'] == 1 else "Will Not Churn"
                            st.metric("Prediction", prediction_text)
                        
                        with col3:
                            risk_class = prediction['risk_level']
                            if risk_class == 'High':
                                st.markdown(f'<p class="risk-high">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                            elif risk_class == 'Medium':
                                st.markdown(f'<p class="risk-medium">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<p class="risk-low">Risk Level: {risk_class}</p>', unsafe_allow_html=True)
                        
                        # Risk visualization
                        fig_risk = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prediction['churn_probability'] * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Risk Score (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},
                                    {'range': [40, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig_risk.update_layout(height=400)
                        st.plotly_chart(fig_risk, use_container_width=True)
            
            else:
                st.info("Please train the model first to make predictions")
        
        with tab3:
            st.subheader("Risk Indicators Analysis")
            
            # High-risk customer identification
            if st.session_state.model_trained:
                # Predict churn for all customers
                predictions = []
                for idx, row in data.iterrows():
                    customer_df = pd.DataFrame([row])
                    pred = st.session_state.predictor.predict_churn(customer_df)
                    predictions.append(pred['churn_probability'])
                
                data_with_predictions = data.copy()
                data_with_predictions['churn_probability'] = predictions
                data_with_predictions['risk_level'] = data_with_predictions['churn_probability'].apply(
                    lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                )
                
                # Risk distribution
                risk_dist = data_with_predictions['risk_level'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_risk_dist = px.pie(
                        values=risk_dist.values,
                        names=risk_dist.index,
                        title="Customer Risk Distribution",
                        color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                    )
                    st.plotly_chart(fig_risk_dist, use_container_width=True)
                
                with col2:
                    # High-risk customers by contract
                    high_risk = data_with_predictions[data_with_predictions['risk_level'] == 'High']
                    if len(high_risk) > 0:
                        high_risk_contract = high_risk['Contract'].value_counts()
                        fig_high_risk = px.bar(
                            x=high_risk_contract.index,
                            y=high_risk_contract.values,
                            title="High-Risk Customers by Contract Type",
                            color=high_risk_contract.values,
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_high_risk, use_container_width=True)
                    else:
                        st.info("No high-risk customers identified")
                
                # Risk factors analysis
                st.subheader("Key Risk Factors")
                
                # Analyze risk factors for high-risk customers
                high_risk_customers = data_with_predictions[data_with_predictions['risk_level'] == 'High']
                
                if len(high_risk_customers) > 0:
                    risk_factors = {
                        'Month-to-month Contract': (high_risk_customers['Contract'] == 'Month-to-month').mean() * 100,
                        'Electronic Check Payment': (high_risk_customers['PaymentMethod'] == 'Electronic check').mean() * 100,
                        'No Online Security': (high_risk_customers['OnlineSecurity'] == 'No').mean() * 100,
                        'Short Tenure (<12 months)': (high_risk_customers['tenure'] < 12).mean() * 100,
                        'High Monthly Charges (>$80)': (high_risk_customers['MonthlyCharges'] > 80).mean() * 100
                    }
                    
                    risk_factors_df = pd.DataFrame(
                        list(risk_factors.items()),
                        columns=['Risk Factor', 'Percentage']
                    )
                    
                    fig_risk_factors = px.bar(
                        risk_factors_df,
                        x='Percentage',
                        y='Risk Factor',
                        orientation='h',
                        title="Risk Factors in High-Risk Customers",
                        color='Percentage',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_risk_factors, use_container_width=True)
                
                # Customer segmentation
                st.subheader("Customer Segmentation")
                
                # Create customer segments based on risk and value
                data_with_predictions['customer_value'] = data_with_predictions['MonthlyCharges'] * data_with_predictions['tenure']
                
                # Define segments
                def get_segment(row):
                    if row['risk_level'] == 'High' and row['customer_value'] > data_with_predictions['customer_value'].quantile(0.7):
                        return 'High Risk - High Value'
                    elif row['risk_level'] == 'High' and row['customer_value'] <= data_with_predictions['customer_value'].quantile(0.7):
                        return 'High Risk - Low Value'
                    elif row['risk_level'] == 'Low' and row['customer_value'] > data_with_predictions['customer_value'].quantile(0.7):
                        return 'Low Risk - High Value'
                    else:
                        return 'Low Risk - Low Value'
                
                data_with_predictions['segment'] = data_with_predictions.apply(get_segment, axis=1)
                
                segment_counts = data_with_predictions['segment'].value_counts()
                
                fig_segments = px.bar(
                    x=segment_counts.index,
                    y=segment_counts.values,
                    title="Customer Segments",
                    color=segment_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_segments.update_layout(xaxis_title="Customer Segment", yaxis_title="Number of Customers")
                st.plotly_chart(fig_segments, use_container_width=True)
            
            else:
                st.info("Please train the model first to analyze risk indicators")
        
        with tab4:
            st.subheader("Customer Details")
            
            # Data table with filters
            st.subheader("Customer Database")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                contract_filter = st.selectbox("Filter by Contract", ["All"] + list(data['Contract'].unique()))
                churn_filter = st.selectbox("Filter by Churn Status", ["All", "Churned", "Not Churned"])
            
            with col2:
                tenure_range = st.slider("Tenure Range (months)", 
                                       int(data['tenure'].min()), 
                                       int(data['tenure'].max()), 
                                       (int(data['tenure'].min()), int(data['tenure'].max())))
                
                charges_range = st.slider("Monthly Charges Range ($)", 
                                        float(data['MonthlyCharges'].min()), 
                                        float(data['MonthlyCharges'].max()), 
                                        (float(data['MonthlyCharges'].min()), float(data['MonthlyCharges'].max())))
            
            with col3:
                if st.session_state.model_trained:
                    risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low"])
                else:
                    risk_filter = "All"
            
            # Apply filters
            filtered_data = data.copy()
            
            if contract_filter != "All":
                filtered_data = filtered_data[filtered_data['Contract'] == contract_filter]
            
            if churn_filter == "Churned":
                filtered_data = filtered_data[filtered_data['Churn'] == 1]
            elif churn_filter == "Not Churned":
                filtered_data = filtered_data[filtered_data['Churn'] == 0]
            
            filtered_data = filtered_data[
                (filtered_data['tenure'] >= tenure_range[0]) & 
                (filtered_data['tenure'] <= tenure_range[1])
            ]
            
            filtered_data = filtered_data[
                (filtered_data['MonthlyCharges'] >= charges_range[0]) & 
                (filtered_data['MonthlyCharges'] <= charges_range[1])
            ]
            
            if st.session_state.model_trained and risk_filter != "All":
                # Add predictions if not already present
                if 'churn_probability' not in filtered_data.columns:
                    predictions = []
                    for idx, row in filtered_data.iterrows():
                        customer_df = pd.DataFrame([row])
                        pred = st.session_state.predictor.predict_churn(customer_df)
                        predictions.append(pred['churn_probability'])
                    filtered_data['churn_probability'] = predictions
                    filtered_data['risk_level'] = filtered_data['churn_probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.4 else 'Low'
                    )
                
                filtered_data = filtered_data[filtered_data['risk_level'] == risk_filter]
            
            # Display filtered data
            st.write(f"Showing {len(filtered_data)} customers")
            
            # Select columns to display
            display_columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                             'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn']
            
            if st.session_state.model_trained and 'churn_probability' in filtered_data.columns:
                display_columns.extend(['churn_probability', 'risk_level'])
            
            st.dataframe(
                filtered_data[display_columns],
                use_container_width=True,
                height=400
            )
            
            # Download option
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_customers_{len(filtered_data)}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👈 Please generate customer data using the sidebar to get started!")
        
        # Show sample of what the app can do
        st.subheader("🎯 What This Dashboard Can Do")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Data Analysis**
            - Generate synthetic customer data
            - Visualize churn patterns by demographics
            - Analyze key risk factors
            - Customer segmentation analysis
            """)
        
        with col2:
            st.markdown("""
            **🤖 Machine Learning**
            - Train Random Forest churn prediction model
            - Individual customer churn prediction
            - Risk level assessment (High/Medium/Low)
            - Feature importance analysis
            """)
        
        st.markdown("""
        **🔍 Interactive Features**
        - Real-time churn probability calculation
        - Customer filtering and segmentation
        - Risk indicator visualization
        - Downloadable reports and data
        """)

if __name__ == "__main__":
    main()
