import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import pickle

# Load data and model
processed_data_path = '/content/drive/MyDrive/hate_speech_detection/data/processed/hate_speech_data_combined.csv'
df = pd.read_csv(processed_data_path)

model_path = '/content/drive/MyDrive/hate_speech_detection/models/logistic_regression_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Predict and visualize
X_test = df['cleaned_text']
y_test = df['label']
y_pred = model.predict(X_test)

st.title('Hate Speech Detection Dashboard')

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
fig = px.imshow(cm, text_auto=True)
st.plotly_chart(fig)
