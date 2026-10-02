import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Model, Scaler, and PCA ---
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
kmeans = joblib.load('model.pkl')

st.title("🧩 Customer Segmentation (PCA + KMeans)")
st.write("Enter customer details to predict their segment based on PCA-reduced clustering (K=3).")

# --- Numeric Inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=5000, value=100)
num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits = st.number_input("Web Visits/Month", min_value=0, max_value=50, value=5)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)
total_purchases = st.number_input("Total Purchases", min_value=0, max_value=200, value=50)
num_catalog_purchases = st.number_input("Catalog Purchases", min_value=0, max_value=50, value=5)
num_deals_purchases = st.number_input("Deals Purchases", min_value=0, max_value=50, value=5)
total_children = st.number_input("Total Children", min_value=0, max_value=10, value=2)
income_per_child = st.number_input("Income per Child", min_value=0, max_value=100000, value=10000)
spending_per_visit = st.number_input("Spending per Visit", min_value=0, max_value=5000, value=200)

# --- Encoded Education ---
education = st.selectbox("Education Level", ['Basic', 'Graduation', 'Master', 'PhD'])
education_basic = 1 if education == 'Basic' else 0
education_graduation = 1 if education == 'Graduation' else 0
education_master = 1 if education == 'Master' else 0
education_phd = 1 if education == 'PhD' else 0

# --- Encoded Marital Status ---
marital_status = st.selectbox(
    "Marital Status",
    ['Alone', 'Divorced', 'Married', 'Single', 'Together', 'Widow', 'YOLO']
)
marital_status_alone = 1 if marital_status == 'Alone' else 0
marital_status_divorced = 1 if marital_status == 'Divorced' else 0
marital_status_married = 1 if marital_status == 'Married' else 0
marital_status_single = 1 if marital_status == 'Single' else 0
marital_status_together = 1 if marital_status == 'Together' else 0
marital_status_widow = 1 if marital_status == 'Widow' else 0
marital_status_yolo = 1 if marital_status == 'YOLO' else 0

# --- Create input DataFrame ---
input_data = pd.DataFrame({
    'Income': [income],
    'Age': [age],
    'Recency': [recency],
    'Total_Children': [total_children],
    'Spending': [total_spending],
    'TotalPurchases': [total_purchases],
    'NumWebVisitsMonth': [num_web_visits],
    'NumWebPurchases': [num_web_purchases],
    'NumCatalogPurchases': [num_catalog_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumDealsPurchases': [num_deals_purchases],
    'Income_per_Child': [income_per_child],
    'Spending_per_Visit': [spending_per_visit],
    'Education_Basic': [education_basic],
    'Education_Graduation': [education_graduation],
    'Education_Master': [education_master],
    'Education_PhD': [education_phd],
    'Marital_Status_Alone': [marital_status_alone],
    'Marital_Status_Divorced': [marital_status_divorced],
    'Marital_Status_Married': [marital_status_married],
    'Marital_Status_Single': [marital_status_single],
    'Marital_Status_Together': [marital_status_together],
    'Marital_Status_Widow': [marital_status_widow],
    'Marital_Status_YOLO': [marital_status_yolo],
})

st.write("### Input Data Preview")
st.dataframe(input_data)

# --- Fix column order to match training ---
expected_columns = [
    'Income', 'Age', 'Recency', 'Total_Children', 'Spending', 'TotalPurchases',
    'NumWebVisitsMonth', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumDealsPurchases', 'Income_per_Child',
    'Spending_per_Visit', 'Education_Graduation', 'Education_Master',
    'Education_PhD', 'Education_Basic', 'Marital_Status_Alone',
    'Marital_Status_Divorced', 'Marital_Status_Married',
    'Marital_Status_Single', 'Marital_Status_Together',
    'Marital_Status_Widow', 'Marital_Status_YOLO'
]

input_data = input_data.reindex(columns=expected_columns, fill_value=0)

# --- Cluster Descriptions ---
cluster_summary = {
    0: " **Cluster 0:** High-income, high-spending loyal customers — premium target segment.",
    1: " **Cluster 1:** Moderate income and spending — balanced, potential growth customers.",
    2: " **Cluster 2:** Low income, low activity — inactive or cost-conscious customers.",
}

# --- Predict Cluster ---
if st.button("🔍 Predict Cluster"):
    scaled_input = scaler.transform(input_data)
    pca_input = pca.transform(scaled_input)
    cluster = kmeans.predict(pca_input)[0]

    st.success(f"Predicted Cluster: {cluster}")
    st.markdown(cluster_summary.get(cluster, "No description available for this cluster."))
