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
    # Drop NA rows for required features
    state_df = state_df.dropna(subset=[
        'cases_import', 'cases_recovered', 'cases_active', 'cases_cluster',
        'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost',
        'daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child',
        'daily_partial_adolescent', 'daily_full_adolescent', 'daily_booster_adolescent', 'daily_booster2_adolescent',
        'daily_partial_adult', 'daily_full_adult', 'daily_booster_adult', 'daily_booster2_adult',
        'daily_partial_elderly', 'daily_full_elderly', 'daily_booster_elderly', 'daily_booster2_elderly',
        'admitted_covid', 'discharged_covid', 'icu_covid', 'vent_covid', 'beds_covid', 'beds_icu_covid',
        'total_child_vax', 'total_adol_vax', 'total_adult_vax', 'total_elderly_vax',
        'cases_lag_1', 'cases_lag_7', 'cases_ma_7'
    ])

    # Select prediction date
    valid_dates = state_df['date'].dt.strftime('%Y-%m-%d').tolist()
    selected_date_str = st.selectbox("Choose a Date for Prediction", valid_dates)
    selected_date = pd.to_datetime(selected_date_str)
    selected_row = state_df[state_df['date'] == selected_date].iloc[0]

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

    features = selected_row[feature_cols].values.reshape(1, -1)

    # Debug output
    st.write(f"✅ Model expects {model.n_features_in_} features")
    st.write(f"📥 Input shape: {features.shape}")
    st.write("🧪 Any NaNs in input:", pd.DataFrame(features, columns=feature_cols).isnull().any().any())
    st.write("🔍 Input Preview:", pd.DataFrame(features, columns=feature_cols))

    try:
        prediction = model.predict(features)[0]
        st.metric("Predicted New Cases (next day)", int(prediction))

        # --- Actual value comparison ---
        next_day = selected_date + pd.Timedelta(days=1)
        next_day_data = state_df[state_df['date'] == next_day]

        if not next_day_data.empty:
            actual_cases = next_day_data['cases_new'].values[0]
            st.metric("📊 Actual Cases (Next Day)", int(actual_cases))
            st.write(f"🧮 Prediction Error: {int(prediction) - int(actual_cases)}")
        else:
            st.info("No actual data available for the next day.")

    except ValueError as e:
        st.error(f"Prediction failed: {e}")
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
