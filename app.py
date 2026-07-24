import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "antara_clean.csv"
    df = pd.read_csv(data_path)
    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["char_count"] = df["content"].str.len()
    df["sentence_count"] = df["content"].str.count(r"[.!?]")
    df["comma_count"] = df["content"].str.count(",")
    df["avg_word_length"] = df["char_count"] / df["content"].str.split().str.len()
    df["word_count"] = df["content"].str.split().str.len()
    df.drop(columns=["char_count"], inplace=True, errors="ignore")
    return df


def main():
    st.set_page_config(layout="wide")
    st.title("Antara News Article Analysis Dashboard")

    df = load_data()

    st.subheader("1. Data Summary")
    st.write("**First 5 rows of the dataset:**")
    st.dataframe(df.head())
    st.write("**Descriptive Statistics:**")
    st.dataframe(df.describe().T)

    st.subheader("2. Exploratory Data Analysis (EDA)")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["word_count"], bins=40, kde=True, ax=ax1)
    ax1.set_title("Distribution of Word Count")
    ax1.set_xlabel("Word Count")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(df["title_length"], bins=30, kde=True, ax=ax2)
    ax2.set_title("Distribution of Title Length")
    ax2.set_xlabel("Characters")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="title_length", y="word_count", data=df, alpha=0.5, ax=ax3)
    ax3.set_title("Title Length vs Word Count")
    ax3.set_xlabel("Title Length")
    ax3.set_ylabel("Word Count")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)

    st.subheader("3. Regression Model & Prediction Results")

    X = df[["title_length", "sentence_count", "avg_word_length", "title_word_count", "comma_count"]]
    y = df["word_count"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("**Actual vs. Predicted Word Count (first 10 samples):**")
    hasil = pd.DataFrame({"Actual": y_test.values, "Prediction": np.round(y_pred, 2)})
    st.dataframe(hasil.head(10))

    fig5, ax5 = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x="Actual", y="Prediction", data=hasil, alpha=0.6, ax=ax5)
    ax5.plot([hasil["Actual"].min(), hasil["Actual"].max()], [hasil["Actual"].min(), hasil["Actual"].max()], color="red", linestyle="--")
    ax5.set_title("Actual vs. Predicted Word Count")
    ax5.set_xlabel("Actual Word Count")
    ax5.set_ylabel("Predicted Word Count")
    ax5.grid(True)
    st.pyplot(fig5)

    st.write("**Model Evaluation Metrics:**")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R-squared (R2): {r2:.2f}")


if __name__ == "__main__":
    if os.environ.get("DASHBOARD_BIGDATA_STREAMLIT_LAUNCHED") != "1":
        os.environ["DASHBOARD_BIGDATA_STREAMLIT_LAUNCHED"] = "1"
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", __file__]))
    main()
