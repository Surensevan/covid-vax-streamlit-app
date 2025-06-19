import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
@st.cache_data
def load_data():
    df = pd.read_csv("merged_levels.csv", parse_dates=['date'])
    demo_df = pd.read_csv("demographic_level_df.csv")
    return df, demo_df

df, demo_df = load_data()

st.title("📊 COVID-19 Vaccination Impact Dashboard - Malaysia")

# Sidebar - State selection
states = df['state'].unique()
selected_state = st.sidebar.selectbox("Select a State", sorted(states))

# Load trained model via upload AFTER state selection
uploaded_model = st.sidebar.file_uploader("📤 Upload Model File (.pkl)", type=["pkl"])

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    model_loaded = True
    st.sidebar.success("✅ Model loaded successfully.")
else:
    model = None
    model_loaded = False
    st.sidebar.warning("⚠️ Please upload 'random_forest_model_better.pkl'.")

# Filter by state
state_df = df[df['state'] == selected_state].copy()

# --- Add lag features (required for prediction) ---
state_df = state_df.sort_values("date")
state_df['cases_lag_1'] = state_df['cases_new'].shift(1)
state_df['cases_lag_7'] = state_df['cases_new'].shift(7)
state_df['cases_ma_7'] = state_df['cases_new'].rolling(window=7).mean()

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

# --- Model Prediction (Updated with exact training features) ---
st.subheader("📈 Model Prediction (Demo)")

if model_loaded:
    state_df = state_df.dropna(subset=[
        'cases_import', 'cases_recovered', 'cases_active', 'cases_cluster',
        'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost',
        'daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child',
        'daily_partial_adolescent', 'daily_full_adolescent', 'daily_booster_adolescent', 'daily_booster2_adolescent',
        'daily_partial_adult', 'daily_full_adult', 'daily_booster_adult', 'daily_booster2_adult',
        'daily_partial_elderly', 'daily_full_elderly', 'daily_booster_elderly', 'daily_booster2_elderly',
        'admitted_covid', 'discharged_covid', 'icu_covid', 'vent_covid', 'beds_covid', 'beds_icu_covid',
        'total_child_vax', 'total_adol_vax', 'total_adult_vax', 'total_elderly_vax', 'MCO',
        'cases_lag_1', 'cases_lag_7', 'cases_ma_7'
    ])

    feature_cols = [
        'cases_import', 'cases_recovered', 'cases_active', 'cases_cluster',
        'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost',
        'daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child',
        'daily_partial_adolescent', 'daily_full_adolescent', 'daily_booster_adolescent', 'daily_booster2_adolescent',
        'daily_partial_adult', 'daily_full_adult', 'daily_booster_adult', 'daily_booster2_adult',
        'daily_partial_elderly', 'daily_full_elderly', 'daily_booster_elderly', 'daily_booster2_elderly',
        'admitted_covid', 'discharged_covid', 'icu_covid', 'vent_covid', 'beds_covid', 'beds_icu_covid',
        'total_child_vax', 'total_adol_vax', 'total_adult_vax', 'total_elderly_vax', 'MCO',
        'cases_lag_1', 'cases_lag_7', 'cases_ma_7'
    ]

    # Select prediction date
    valid_dates = state_df[state_df['date'].isin(
        df[df['state'] == selected_state]['date'] - pd.Timedelta(days=1)
    )]['date'].dt.strftime('%Y-%m-%d').tolist()
    selected_date_str = st.selectbox("Choose a Date for Prediction", valid_dates)
    selected_date = pd.to_datetime(selected_date_str)
    selected_row = state_df[state_df['date'] == selected_date].iloc[0]

    features = selected_row[feature_cols].values.reshape(1, -1)

    prediction = model.predict(features)[0]
    st.metric("Predicted New Cases (next day)", int(prediction))

    # Evaluation block (manual sample based on selected row)
    st.subheader("🧪 Model Evaluation Summary")
    y_true = state_df['cases_new'].shift(-1).dropna()
    X_eval = state_df[feature_cols].iloc[:-1]
    y_pred = model.predict(X_eval)

    st.write(f"**MAE:** {mean_absolute_error(y_true, y_pred):.2f}")
    st.write(f"**RMSE:** {mean_squared_error(y_true, y_pred, squared=False):.2f}")
    st.write(f"**R² Score:** {r2_score(y_true, y_pred):.3f}")
else:
    st.warning("Model file not loaded. Please upload 'random_forest_model_better.pkl' in the sidebar.")

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
