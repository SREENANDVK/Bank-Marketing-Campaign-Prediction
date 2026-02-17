import streamlit as st
import joblib
import pickle
from PIL import Image
import pandas as pd

model = joblib.load("random_forest_model.pkl")
ohe = joblib.load("onehot_encoder.pkl")
feature_cols = joblib.load("model_features.pkl")

# Categorical columns (must match training)
cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

def main():
    st.title(':blue[Bank Marketing Campaign Prediction]')
    image = Image.open('bank.jpg')
    st.image(image, width=800)

    age = st.number_input("Age", min_value=0, step=1)
    job = st.selectbox("Job", ['management','technician','entrepreneur','blue-collar','unknown','retired','admin.','services','self-employed','unemployed','housemaid','student'])
    marital = st.selectbox("Marital Status", ["divorced","married","single"])
    education = st.selectbox("Education", ["primary","secondary","tertiary","unknown"])
    default = st.selectbox("Has Credit in Default?", ["no","yes"])
    balance = st.number_input("Average Yearly Balance")
    housing = st.selectbox("Has Housing Loan?", ["no","yes"])
    loan = st.selectbox("Has Personal Loan?", ["no","yes"])
    contact = st.selectbox("Contact Type", ["cellular","telephone","unknown"])
    day = st.number_input("Last Contact Day", min_value=1, max_value=31)
    month = st.selectbox("Last Contact Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
    campaign = st.number_input("Contacts During Campaign", min_value=1)
    pdays = st.number_input("Days Since Last Contact", min_value=-1)
    previous = st.number_input("Previous Contacts", min_value=0)
    poutcome = st.selectbox("Previous Campaign Outcome", ["failure","unknown","success","other"])

    if st.button("Predict"):

        default = 1 if default == "yes" else 0
        housing = 1 if housing == "yes" else 0
        loan = 1 if loan == "yes" else 0

        input_df = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'default': default,
            'balance': balance,
            'housing': housing,
            'loan': loan,
            'contact': contact,
            'day': day,
            'month': month,
            'campaign': campaign,
            'pdays': pdays,
            'previous': previous,
            'poutcome': poutcome
        }])

        encoded = ohe.transform(input_df[cat_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(cat_cols)
        )

        final_input = pd.concat(
            [input_df.drop(columns=cat_cols), encoded_df],
            axis=1
        )

        final_input = final_input.reindex(columns=feature_cols, fill_value=0)

        pred = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0][1]

        if pred == 1:
            st.success(f"✅ Likely to Subscribe (Probability: {prob:.2%})")
        else:
            st.error(f"❌ Not Likely to Subscribe (Probability: {prob:.2%})")



if __name__ == "__main__":
    main()

