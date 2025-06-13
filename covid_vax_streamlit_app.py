import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load datasets
@st.cache_data
def load_data():
    df = pd.read_csv("merged_levels.csv", parse_dates=['date'])
    demo_df = pd.read_csv("demographic_level_df.csv")
    return df, demo_df

df, demo_df = load_data()

# Load trained model
try:
    model = joblib.load("random_forest_model.pkl")
    model_loaded = True
except FileNotFoundError:
    model = None
    model_loaded = False

st.title("📊 COVID-19 Vaccination Impact Dashboard - Malaysia")

# Sidebar - State selection
states = df['state'].unique()
selected_state = st.sidebar.selectbox("Select a State", sorted(states))

# Filter by state
state_df = df[df['state'] == selected_state].copy()

# --- Chart: Daily New Cases ---
st.subheader(f"🦠 Daily New COVID-19 Cases - {selected_state}")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(state_df['date'], state_df['cases_new'], label='New Cases', color='red')
ax1.set_xlabel("Date")
ax1.set_ylabel("Cases")
ax1.set_title("Daily New Cases")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# --- Chart: Cumulative Vaccination ---
st.subheader(f"💉 Cumulative Full Adult Vaccination - {selected_state}")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(state_df['date'], state_df['daily_full_adult'].cumsum(), label='Full Adult Vax', color='green')
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Count")
ax2.set_title("Vaccination Progress")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# --- Model Prediction (Placeholder) ---
st.subheader("📈 Model Prediction (Demo)")

if model_loaded:
    # Example: Use the latest available features to make prediction
    # You should customize this based on actual feature engineering used in training
    latest_row = state_df.dropna().iloc[-1]
    features = [
        latest_row['daily_full_adult'],
        latest_row['daily_partial_adult'],
        latest_row['cases_active'],
        latest_row['cases_recovered'],
    ]
    prediction = model.predict([features])[0]
    st.metric("Predicted New Cases (next day)", int(prediction))
else:
    st.warning("Model file not found. Please ensure 'random_forest_model.pkl' is in the folder.")

# --- Demographic Info ---
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Demographic Snapshot")
if selected_state in demo_df['state'].values:
    state_info = demo_df[demo_df['state'] == selected_state].iloc[0]
    st.sidebar.write(f"**Population**: {state_info['population']:.0f}k")
    st.sidebar.write(f"**Mean Income**: RM {state_info['income_mean']:.2f}")
    st.sidebar.write(f"**Median Income**: RM {state_info['income_median']:.2f}")
else:
    st.sidebar.write("No demographic data available.")
