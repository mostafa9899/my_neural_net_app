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

        # 3) Load your saved artifacts (model, scaler, label encoder)
        model = load_model("model.h5")
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # 4) Preprocess
        # Drop "Name" and "DISEASE" from the features to be predicted
        df_for_pred = df.drop(columns=["Name", "DISEASE"], errors="ignore")
        df_for_pred = df_for_pred.dropna()
        X_scaled = scaler.transform(df_for_pred)

        # 5) Predict
        preds = model.predict(X_scaled)
        pred_classes = np.argmax(preds, axis=1)
        decoded_labels = label_encoder.inverse_transform(pred_classes)

        # 6) Display results
        # Add the predictions to the original DataFrame
        df["PREDICTION"] = np.nan
        df.loc[df_for_pred.index, "PREDICTION"] = decoded_labels

        # Only display "Name" and "PREDICTION" if "Name" exists; otherwise, just "PREDICTION"
        if "Name" in df.columns:
            result_df = df[["Name", "PREDICTION"]]
        else:
            result_df = df[["PREDICTION"]]

        st.write("Predictions:")
        st.dataframe(result_df)

        # Provide a download link for the results as an Excel file
        st.markdown(get_table_download_link(result_df), unsafe_allow_html=True)

def get_table_download_link(df):
    """
    Generates a link to download the dataframe as an Excel file.
    """
    import base64
    import io
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    return f'<a href="data:file/excel;base64,{b64}" download="predictions.xlsx">Download Predictions Excel file</a>'

if __name__ == "__main__":
    main()

