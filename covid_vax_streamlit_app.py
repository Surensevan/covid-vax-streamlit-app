import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define relevant features globally
feature_cols = [
    'cases_import', 'cases_recovered', 'cases_active', 'cases_cluster',
    'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost',
    'daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child',
    'daily_partial_adolescent', 'daily_full_adolescent', 'daily_booster_adolescent', 'daily_booster2_adolescent',
    'daily_partial_adult', 'daily_full_adult', 'daily_booster_adult', 'daily_booster2_adult',
    'daily_partial_elderly', 'daily_full_elderly', 'daily_booster_elderly', 'daily_booster2_elderly',
    'admitted_covid', 'discharged_covid', 'icu_covid', 'vent_covid', 'beds_covid', 'beds_icu_covid',
    'total_child_vax', 'total_adol_vax', 'total_adult_vax', 'total_elderly_vax', 'MCO'
]

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("merged_levels.csv", parse_dates=['date'])
    df = df.sort_values("date")

    # Add lag features after loading
    df['cases_lag_1'] = df['cases_new'].shift(1)
    df['cases_lag_7'] = df['cases_new'].shift(7)
    df['cases_lag_14'] = df['cases_new'].shift(14)
    df['cases_ma_7'] = df['cases_new'].rolling(window=7).mean()

    lag_features = ['cases_lag_1', 'cases_lag_7', 'cases_lag_14', 'cases_ma_7']
    all_features = feature_cols + lag_features

    expected_columns = set(all_features + ['cases_new', 'date'])
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        st.error(f"❌ Missing columns in data: {missing_columns}")
        st.stop()
    df = df[[*all_features, 'cases_new', 'date']].copy()
    return df

df = load_data()

st.title("📊 COVID-19 Vaccination Impact Dashboard - Malaysia")

# --- Model Upload ---
uploaded_model = st.file_uploader("📤 Upload Model File (.pkl)", type=["pkl"])

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    model_loaded = True
    st.success("✅ Model loaded successfully.")
else:
    model = None
    model_loaded = False
    st.warning("⚠️ Please upload 'random_forest_model_better.pkl'.")

# --- Chart: Daily New Cases ---
st.subheader("🦠 Daily New COVID-19 Cases")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df['date'], df['cases_new'], label='New Cases', color='red')
ax1.set_xlabel("Date")
ax1.set_ylabel("Cases")
ax1.set_title("Daily New Cases")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# --- Chart: Cumulative Vaccination ---
st.subheader("💉 Cumulative Full Adult Vaccination")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df['date'], df['daily_full_adult'].cumsum(), label='Full Adult Vax', color='green')
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Count")
ax2.set_title("Vaccination Progress")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# --- Model Prediction ---
st.subheader("📈 Model Prediction")

# --- Date Selection (Dropdown without state filter) ---
date_options = df['date'].dropna().dt.date.unique()
selected_date = st.selectbox("Select Date for Prediction", options=sorted(date_options))

if model_loaded and selected_date:
    df = df.dropna(subset=feature_cols + ['cases_lag_1', 'cases_lag_7', 'cases_lag_14', 'cases_ma_7', 'cases_new'])

    st.markdown(f"---\n📅 **Date:** {selected_date}")
    row = df[df['date'].dt.date == selected_date]

    if not row.empty:
        try:
            all_features = feature_cols + ['cases_lag_1', 'cases_lag_7', 'cases_lag_14', 'cases_ma_7']
            features = row[all_features].values.reshape(1, -1)
            prediction = model.predict(features)[0]
            st.metric("Predicted New Cases (next day)", int(prediction))
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    else:
        st.warning(f"⚠️ No valid data for selected date: {selected_date}")

    # Evaluation block
    st.subheader("🧪 Model Evaluation Summary")
    try:
        all_features = feature_cols + ['cases_lag_1', 'cases_lag_7', 'cases_lag_14', 'cases_ma_7']
        df_eval = df.dropna(subset=all_features + ['cases_new']).copy()
        X_eval = df_eval[all_features]
        y_true = df_eval['cases_new'].shift(-1).dropna()
        X_eval = X_eval.loc[y_true.index]

        y_pred = model.predict(X_eval)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        st.write("**MAE:** {:.2f}".format(mae))
        st.write("**RMSE:** {:.2f}".format(rmse))
        st.write("**R² Score:** {:.3f}".format(r2))
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info("Please select a date and upload a valid model.")
