import streamlit as st 
import requests 


st.set_page_config(
    page_title="Heart Diseases Prediction ",
    page_icon="🏥",
    layout="wide"
)

st.title("Heart Diseases Prediction",text_alignment="center",width="stretch")
st.caption("Predict can a person have heart diseases or not",text_alignment="center")

API_URL = "http://127.0.0.1:8000"

with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_check = requests.get(API_URL)
        if health_check.status_code == 200:
            st.success("🟢 Backend API Online")
        else:
            st.warning("🟡 Backend API: Degraded")
    except requests.exceptions.Exception:
        st.warning("🟡 Backend API: Waking Up / Idle")
        st.caption("Free-tier servers sleep after 15 minutes of inactivity. It takes ~40 seconds to wake up on the first request. Your app will work normally when you click Predict!")
        

st.markdown("### Patient Details ")
st.write("Adjust the parameters below according to patient data")


col1,col2,col3 = st.columns(3)

with col1:
    st.subheader("patient data")

    Age = st.number_input("Age", max_value=100, min_value=0, value=55)

    Sex = st.selectbox("Sex",["M","F"])

    ChestPainType = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])

    RestingBP = st.number_input("Resting BP", min_value=30, max_value=200, value=140)

with col2:
    st.subheader("Clinical measurements")

    Cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=240)

    FastingBS = st.number_input("Fasting Blood Sugar", min_value=0, max_value=1, value=0)

    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

    MaxHR = st.number_input("Maximum Heart Rate", min_value=40, max_value=220, value=150)

with col3:
    st.subheader("Exercise and ECG")

    ExerciseAngina = st.selectbox("Exercise Angina", ["Y", "N"])

    Oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

    ST_Slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

st.markdown("---")

if st.button("Predict Heart Diseases", use_container_width=True, type="primary"):
    payload = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope,
    }
    with st.spinner("Calculating patient data and predicting result"):
        try:

            response = requests.post(f"{API_URL}/predict",json=payload,timeout = 10)

            if response.status_code == 200:
                data = response.json()
                result = data["data"]
                if result == 1:
                    st.warning("Heart Disease")
                else:
                    st.success("No Heart Disease")
                
            else:
                st.error(f"Prediction failed: {response.text}")
        except requests.exceptions.Timeout:
            st.error("Request Time out backend server take too long (60 sec)")
        except requests.exceptions.RequestException:
            st.error("Unable to connect Backend API ")



