import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def main():
    st.title("Neural Network Prediction App")
    st.write("Upload an Excel file to get predictions.")

    # 1) Upload file
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # 2) Read the file into a DataFrame
        df = pd.read_excel(uploaded_file)

        st.write("Preview of uploaded file:")
        st.dataframe(df.head())

        # 3) Load your saved artifacts (model, scaler, label encoder)
        model = load_model("model.h5")

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # If you had an imputer or other transforms, load them similarly:
        # with open("imputer.pkl", "rb") as f:
        #     imputer = pickle.load(f)
        
        # 4) Preprocess
        # Drop or ignore columns as needed (like "Name", "DISEASE" if present):
        df_for_pred = df.drop(columns=["Name", "DISEASE"], errors="ignore")

        # Drop or fill missing data (example):
        df_for_pred = df_for_pred.dropna()  # or fill with zeros, up to you

        # Scale
        X_scaled = scaler.transform(df_for_pred)

        # 5) Predict
        preds = model.predict(X_scaled)
        pred_classes = np.argmax(preds, axis=1)
        decoded_labels = label_encoder.inverse_transform(pred_classes)

        # 6) Display results
        # Create a new column in original df (for demonstration)
        df["PREDICTION"] = np.nan  # default
        df.loc[df_for_pred.index, "PREDICTION"] = decoded_labels

        st.write("Predictions:")
        st.dataframe(df)

        # Optionally provide a download link for the results as a new Excel file:
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

def get_table_download_link(df):
    """
    Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    import base64
    import io
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False)  # write to BytesIO buffer
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    return f'<a href="data:file/excel;base64,{b64}" download="predictions.xlsx">Download Predictions Excel file</a>'

if __name__ == "__main__":
    main()
