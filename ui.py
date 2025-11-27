import streamlit as st
import pandas as pd
import io
from shift import run_shift_planning   # <-- uses your backend functions

st.set_page_config(page_title="Shift Planner AI", layout="wide")
st.title("🚦 AI Shift Planning – 6-Month Optimizer")
st.write("Upload your Excel sheet and generate an optimized shift-plan with ML.")

uploaded = st.file_uploader("Upload your Zones Excel (.xlsx)", type="xlsx")

if uploaded:
    df = pd.read_excel(uploaded)
    st.success("File uploaded successfully!")
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    if st.button("Run Shift Optimization 🚀"):

        with st.spinner("Training ML model and calculating assignments…"):

            try:
                final_df, metrics = run_shift_planning(df)
            except Exception as e:
                st.error(f"❌ Error: {e}")
            else:
                st.success("🎉 Shift Plan Generated Successfully!")

                # METRICS
                st.subheader("📊 Model Performance")
                col1, col2 = st.columns(2)
                col1.metric("MAE", f"{metrics['mae']:.2f}")
                col2.metric("Accuracy", f"{metrics['accuracy_pct']:.2f}%")

                # PREVIEW OUTPUT
                st.subheader("📋 Final Shift Plan Preview")
                st.dataframe(final_df.head())

                # DOWNLOAD FILE
                buffer = io.BytesIO()
                final_df.to_excel(buffer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="📥 Download Shift Plan Excel",
                    data=buffer,
                    file_name="Shift_Plan_Final_6Months.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:
    st.info("Upload an Excel file to begin.")
