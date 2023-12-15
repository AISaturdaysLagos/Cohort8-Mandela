import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd

# Streamlit app title
st.title('Anomaly Detection using KNN')

# Upload dataset
uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Feature selection
    features = df[['C6H6(GT)', 'CO(GT)', 'NOx(GT)', 'NO2(GT)']]

    # Standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Anomaly Detection
    threshold = 0.03
    df['anomaly'] = (features_scaled.mean(axis=1) > threshold).astype(int)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, df['anomaly'], test_size=0.2, random_state=42)

    # KNN Model Creation and Training
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Predictions on the entire dataset
    df['predicted_anomaly'] = knn_model.predict(features_scaled)

    # Display Results
    st.subheader('Original Dataset')
    st.write(df)

    st.subheader('Results')
    st.write("Classification Report:")
    st.write(classification_report(df['anomaly'], df['predicted_anomaly']))

    # Display Anomaly Indices
    st.subheader('Anomaly Indices:')
    anomaly_indices = df.index[df['predicted_anomaly'] == 1]
    st.write(anomaly_indices)
