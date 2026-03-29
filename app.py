import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Stats Analyzer", layout="wide")
st.title("📊 ANOVA & Post-Hoc Tukey HSD App")

# --- Upload Dataset ---
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin1')
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Column Selection ---
    col1, col2 = st.columns(2)
    with col1:
        group_col = st.selectbox("Select Groups (Categorical)", df.columns)
    with col2:
        value_col = st.selectbox("Select Values (Numerical)", df.columns)

    if st.button("Run Statistical Analysis"):
        st.divider()
        # 1. ANOVA
        groups = [group[1] for group in df.groupby(group_col)[value_col]]
        f_stat, p_val = stats.f_oneway(*groups)

        m1, m2 = st.columns(2)
        m1.metric("F-Statistic", round(f_stat, 4))
        m2.metric("P-Value", round(p_val, 4))

        if p_val < 0.05:
            st.success("Significant difference found!")
            # 2. Tukey HSD
            tukey = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=0.05)
            st.subheader("Tukey HSD Test Results")
            st.text(str(tukey.summary()))
            
            # 3. Plot
            fig, ax = plt.subplots()
            sns.boxplot(x=group_col, y=value_col, data=df, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No significant difference found.")