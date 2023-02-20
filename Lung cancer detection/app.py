import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
import joblib

st.header("Lung Cancer Detection")
df = pd.read_csv("Data.csv")

X  = df[["GENDER","AGE","SMOKING","YELLOW FINGERS","ANXIETY","PEER PRESSURE","CHRONIC DISEASE","FATIGUE","ALLERGY","WHEEZING","ALCOHOL CONSUMPTION","COUGHING","SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]]
y = df["DETECTION RESULT"]

clf = LogisticRegression()
clf.fit(X, y)
joblib.dump(clf, "model.pkl")

a = st.number_input("GENDER: Enter 1 for Male and 0 for Female", min_value=0, max_value=1)
b = st.slider("AGE: Enter your Age", min_value=1, max_value=100)
c = st.number_input("SMOKING: Enter 1 if you smoke or 0 if you don't smoke", min_value=0, max_value=1)
d = st.number_input("YELLOW FINGERS: Enter 1 if you have yellow fingers or 0 if you don't", min_value=0, max_value=1)
e = st.number_input("ANXIETY: Enter 1 if you have anxiety and 0 if you don't", min_value=0, max_value=1)
f = st.number_input("PEER PRESSURE: Enter 1 if you feel you suffer from peer pressure or 0 if you don't", min_value=0, max_value=1)
g = st.number_input("CHRONIC DISEASE: Enter 1 if you suffer from a chronic disease or O if you don't", min_value=0, max_value=1)
h = st.number_input("FATIGUE: Enter 1 if you have fatigue or 0 if you don't", min_value=0, max_value=1)
i = st.number_input("ALLERGY: Enter 1 if you have some sort of allergy or 0 if you don't", min_value=0, max_value=1)
j = st.number_input("WHEEZING: Enter 1 if you wheeze or 0 if you don't", min_value=0, max_value=1)
k =  st.number_input("ALCOHOL CONSUMPTION: Enter 1 if you consume alcohol or 0 if you don't", min_value=0, max_value=1)
l = st.number_input("COUGHING: Enter 1 if you cough a lot or 0 if you don't", min_value=0, max_value=1)
m = st.number_input("SHORTNESS OF BREATH: Enter 1 if you suffer from shortness of breath or 0 if you don't", min_value=0, max_value=1)
n =  st.number_input("SWALLOWING DIFFICULTY: Enter 1 if you have difficulty swallowing or 0 if you don't", min_value=0, max_value=1)
o =  st.number_input("CHEST PAIN: Enter 1 if you have chest pain or 0 if you don't", min_value=0, max_value=1)

if st.button("Submit"):
    # Unpickle classifier
    clf = joblib.load("model.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o]],
                     columns = ["GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC_DISEASE","FATIGUE","ALLERGY","WHEEZING","ALCOHOL_CONSUMPTION","COUGHING","SHORTNESS_OF_BREATH","SWALLOWING_DIFFICULTY","CHEST_PAIN"])
    
    # Get Prediction 
    prediction = clf.predict(X)[0]
    
    # Output Prediction
    st.text(f"{prediction}")
    
        
