# ========== Imports ==========
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ========== Page Setup ==========
st.set_page_config(page_title="Molecular Biology ML App", layout="wide")

# ========== Sidebar Navigation ==========
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio("Go to", [
    "📁 Upload & Explore",
    "🔍 Clustering",
    "🧠 Train Model",
    "📈 Predict",
    "👤 Creator Info"
])

# ========== Main Content Based on Selected Section ==========
if section == "📁 Upload & Explore":
    st.title("📁 Upload & Explore Your Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📋 Preview of Dataset")
        st.dataframe(df)

        search_term = st.text_input("🔎 Filter rows (by label/class):")
        if search_term:
            df = df[df[df.columns[-1]].astype(str).str.contains(search_term, case=False, na=False)]
            st.write("Filtered Data:")
            st.dataframe(df)

        st.subheader("📊 Summary Statistics")
        st.write(df.describe())

        st.subheader("📈 Scatter Plot")
        if len(df.columns) >= 2:
            x_axis = st.selectbox("X-axis", df.columns[:-1])
            y_axis = st.selectbox("Y-axis", df.columns[:-1], index=1)
            color_col = df.columns[-1]
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
            st.plotly_chart(fig)

elif section == "🔍 Clustering":
    st.title("Clustering with KMeans")

    uploaded_file = st.file_uploader("Upload a CSV or TXT file for clustering", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file, sep='\t')

        st.write("### Uploaded Data")
        st.dataframe(df)

        # Drop last column if it's labels
        features = df.iloc[:, :-1]

        # Sidebar for number of clusters
        k = st.sidebar.slider("Number of clusters (K)", 2, 10, 3)

        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features)

        st.write("### Clustering Results")
        st.dataframe(df)

        # Optional: show plot
        st.subheader("Cluster Scatter Plot")
        if features.shape[1] >= 2:
            x_axis = st.selectbox("X-axis", features.columns)
            y_axis = st.selectbox("Y-axis", features.columns, index=1)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=df['Cluster'].astype(str), title="KMeans Clusters")
            st.plotly_chart(fig)


elif section == "🧠 Train Model":
    st.title("Train a Classification Model")

    uploaded_file = st.file_uploader("Upload a CSV or TXT file for training", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file, sep='\t')

        st.write("### Uploaded Data")
        st.dataframe(df)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"Model Accuracy: {acc:.2f}")


elif section == "📈 Predict":
    st.title("Load Model & Predict")

    uploaded_model = st.file_uploader("Upload a trained model (.pkl)", type=["pkl"])
    uploaded_test = st.file_uploader("Upload a test dataset (CSV)", type=["csv"])

    if uploaded_model and uploaded_test:
        model = joblib.load(uploaded_model)
        test_df = pd.read_csv(uploaded_test)

        st.write("### Test Data")
        st.dataframe(test_df)

        expected_features = model.feature_names_in_
        test_filtered = test_df[expected_features]
        predictions = model.predict(test_filtered)

        # Add predictions to DataFrame
        test_df["Predicted Class"] = predictions

        st.subheader("Predictions")
        st.dataframe(test_df)

        # OPTIONAL: if 'actual' label exists, compare
        if "Actual" in test_df.columns or test_df.columns[-2] not in expected_features:
            # Try to detect actual labels
            actual_col = test_df.columns[-2]
            actual = test_df[actual_col]
            accuracy = accuracy_score(actual, predictions)
            st.success(f"Prediction Accuracy vs. Actual: {accuracy:.2f}")

            # Show comparison chart
            st.subheader("Prediction vs Actual")
            comp_df = pd.DataFrame({
                "Actual": actual,
                "Predicted": predictions
            })
            st.dataframe(comp_df)

            # Optional bar chart
            fig = px.histogram(comp_df, barmode="group")
            st.plotly_chart(fig)

elif section == "👤 Creator Info":
    st.title("👤 Creator Info")
    st.markdown("**Name:** ΜΠΡΑΧΜΑ ΤΖΩΡΤΖ ΣΑΒΑΔΡΑΜ")
    st.markdown("**Student ID:** inf2022176")
    st.markdown("**School-Institute:** Ionian university - Informatics Department")
