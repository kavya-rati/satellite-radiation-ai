import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Satellite Radiation AI", layout="wide")

st.title("AI-Based Satellite Radiation Failure Detection")
st.caption("AI-powered subsystem reliability monitoring")

# Sample dataset
sample_data = pd.DataFrame({
    "dose":[0,10,20,30,40,50,60,70,80,90],
    "voltage":[5.1,4.9,4.7,4.5,4.2,3.8,3.4,2.9,2.4,1.9],
    "current":[0.50,0.52,0.55,0.58,0.60,0.65,0.70,0.74,0.78,0.82]
})

st.subheader("Download Sample Dataset")

st.download_button(
    label="Download Sample Radiation Dataset",
    data=sample_data.to_csv(index=False),
    file_name="radiation_sample_data.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader("Upload radiation dataset (CSV)", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    initial_voltage = df["voltage"].iloc[0]
    final_voltage = df["voltage"].iloc[-1]
    voltage_drop = ((initial_voltage - final_voltage) / initial_voltage) * 100

    avg_voltage = df["voltage"].mean()
    max_current = df["current"].max()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Initial Voltage", f"{initial_voltage:.2f} V")
    col2.metric("Final Voltage", f"{final_voltage:.2f} V")
    col3.metric("Voltage Drop", f"{voltage_drop:.2f}%")
    col4.metric("Max Current", f"{max_current:.2f} A")

    model = IsolationForest(contamination=0.2, random_state=42)
    df["anomaly"] = model.fit_predict(df[["voltage","current"]])

    anomalies = df[df["anomaly"] == -1]

    st.subheader("AI Anomaly Detection")

    if len(anomalies) > 0:
        st.warning(f"{len(anomalies)} abnormal readings detected")
        st.dataframe(anomalies)
    else:
        st.success("No anomalies detected")

    st.subheader("AI Failure Risk Assessment")

    anomaly_count = len(anomalies)
    risk_score = (voltage_drop * 0.6) + (anomaly_count * 10)
    risk_score = min(risk_score,100)

    col1,col2 = st.columns(2)

    col1.metric("AI Risk Score",f"{risk_score:.1f}%")

    if risk_score < 30:
        col2.success("Low Failure Risk")
    elif risk_score < 60:
        col2.warning("Moderate Risk")
    else:
        col2.error("High Failure Risk")

    st.subheader("Radiation Behavior Analysis")

    col1,col2 = st.columns(2)

    fig1 = px.line(df,x="dose",y="voltage",markers=True,title="Radiation Dose vs Voltage")
    fig1.add_hline(y=3.0,line_dash="dash",annotation_text="Failure Threshold")

    col1.plotly_chart(fig1,use_container_width=True)

    fig2 = px.line(df,x="dose",y="current",markers=True,title="Radiation Dose vs Current")

    col2.plotly_chart(fig2,use_container_width=True)

    st.subheader("AI System Recommendation")

    if df["voltage"].min() < 3.0:
        st.error("Critical voltage degradation detected. Subsystem requires validation.")
    else:
        st.success("Subsystem operating within safe radiation limits.")