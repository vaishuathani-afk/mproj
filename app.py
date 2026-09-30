import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare AI IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1220;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 32px;
    font-weight: bold;
}

.subtitle {
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 25px;
}

.card {
    background-color: #1e293b;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #334155;
}

.card-title {
    font-size: 16px;
    color: #cbd5e1;
}

.card-value {
    font-size: 30px;
    font-weight: bold;
    color: #38bdf8;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ============================================================
# LOGIN
# ============================================================

if not st.session_state.logged_in:

    st.markdown(
        "<h1 style='text-align:center;'>🛡️ Healthcare AI IDS</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#9ca3af;'>"
        "AI-Based Healthcare Intrusion Detection System"
        "</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.markdown("### 🔐 Secure Login")

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button(
            "Login",
            use_container_width=True
        ):

            if username == "admin" and password == "1234":

                st.session_state.logged_in = True
                st.rerun()

            else:

                st.error(
                    "Invalid username or password"
                )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ Healthcare AI IDS")

    st.markdown("---")

    if st.button(
        "🏠 Dashboard",
        use_container_width=True
    ):
        st.session_state.page = "Dashboard"

    if st.button(
        "📂 Dataset Analysis",
        use_container_width=True
    ):
        st.session_state.page = "Dataset Analysis"

    if st.button(
        "🤖 AI Models",
        use_container_width=True
    ):
        st.session_state.page = "AI Models"

    if st.button(
        "🚨 Intrusion Detection",
        use_container_width=True
    ):
        st.session_state.page = "Intrusion Detection"

    if st.button(
        "⚠️ Risk Analysis",
        use_container_width=True
    ):
        st.session_state.page = "Risk Analysis"

    if st.button(
        "📊 Model Comparison",
        use_container_width=True
    ):
        st.session_state.page = "Model Comparison"

    if st.button(
        "🔢 Confusion Matrix",
        use_container_width=True
    ):
        st.session_state.page = "Confusion Matrix"

    if st.button(
        "📋 Prediction Results",
        use_container_width=True
    ):
        st.session_state.page = "Prediction Results"

    st.markdown("---")

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div class='main-title'>"
    "🛡️ Healthcare AI Intrusion Detection System"
    "</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>"
    "Machine Learning based security monitoring "
    "for healthcare access activity"
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# DATASET UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload Healthcare Dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is None:

    st.info(
        "Upload your Healthcare IDS CSV or XLSX dataset."
    )

    st.markdown("""
### Expected columns

- Patient_ID
- Doctor_ID
- User_Role
- Department
- Access_Type
- Login_Attempts
- Access_Time
- Device_Type
- Location
- Patient_Record_Accessed
- Label

**Label:**  
0 = Normal  
1 = Attack
""")

    st.stop()


# ============================================================
# READ DATA
# ============================================================

try:

    if uploaded_file.name.endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

except Exception as e:

    st.error("Unable to read the dataset.")
    st.error(str(e))
    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

df = df.dropna(
    how="all"
)

df = df.dropna(
    axis=1,
    how="all"
)


# ============================================================
# CHECK LABEL
# ============================================================

if "Label" not in df.columns:

    st.error(
        "Your dataset must contain a 'Label' column."
    )

    st.stop()


# ============================================================
# TARGET
# ============================================================

y = df["Label"].copy()


# Convert text labels
if y.dtype == "object":

    label_map = {
        "Normal": 0,
        "Attack": 1,
        "normal": 0,
        "attack": 1,
        "NORMAL": 0,
        "ATTACK": 1
    }

    y = y.map(label_map)

    if y.isna().any():

        unique_values = (
            df["Label"]
            .dropna()
            .unique()
        )

        if len(unique_values) == 2:

            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            y = df["Label"].map(
                mapping
            )

        else:

            st.error(
                "Label must contain Normal/Attack "
                "or 0/1."
            )

            st.stop()


y = pd.to_numeric(
    y,
    errors="coerce"
)

valid_rows = y.notna()

df = df.loc[
    valid_rows
].copy()

y = y.loc[
    valid_rows
].astype(int)


if not set(
    y.unique()
).issubset({0, 1}):

    st.error(
        "Label should contain only 0/1 "
        "or Normal/Attack."
    )

    st.stop()


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col not in [
        "Label",
        "Patient_ID"
    ]
]

X = df[
    feature_columns
].copy()


# Remove constant columns
constant_columns = [
    col
    for col in X.columns
    if X[col].nunique(
        dropna=False
    ) <= 1
]

if constant_columns:

    X = X.drop(
        columns=constant_columns
    )


# ============================================================
# NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()

categorical_features = [
    col
    for col in X.columns
    if col not in numeric_features
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42
    )


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Support Vector Machine":
        SVC(
            random_state=42
        )
}


# ============================================================
# TRAIN MODELS
# ============================================================

results = {}

trained_models = {}

predictions = {}


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    pred = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    results[name] = (
        accuracy * 100
    )

    trained_models[name] = pipeline

    predictions[name] = pred


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=results.get
)

best_model = trained_models[
    best_model_name
]

best_predictions = predictions[
    best_model_name
]

best_accuracy = results[
    best_model_name
]


# ============================================================
# TEST RESULTS
# ============================================================

test_results = df.loc[
    X_test.index
].copy()

test_results["Actual"] = (
    y_test.values
)

test_results["Predicted"] = (
    best_predictions
)


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(row):

    score = 0

    # Login attempts
    if "Login_Attempts" in row.index:

        try:

            login_attempts = float(
                row["Login_Attempts"]
            )

            if login_attempts >= 6:
                score += 2

            elif login_attempts >= 4:
                score += 1

        except:

            pass


    # Unknown device
    if "Device_Type" in row.index:

        device = str(
            row["Device_Type"]
        ).lower()

        if device in [
            "unknown",
            "unknown device"
        ]:

            score += 2


    # Unknown user role
    if "User_Role" in row.index:

        role = str(
            row["User_Role"]
        ).lower()

        if role == "unknown":

            score += 1


    # Outside location
    if "Location" in row.index:

        location = str(
            row["Location"]
        ).lower()

        if location == "outside":

            score += 1


    # Delete access
    if "Access_Type" in row.index:

        access_type = str(
            row["Access_Type"]
        ).lower()

        if access_type == "delete":

            score += 1


    # Final risk
    if score >= 4:

        return "High"

    elif score >= 2:

        return "Medium"

    else:

        return "Low"


test_results["Risk_Level"] = (
    test_results.apply(
        calculate_risk,
        axis=1
    )
)


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        "## 🏠 Security Dashboard"
    )

    predicted_attacks = int(
        (best_predictions == 1).sum()
    )

    normal_count = int(
        (best_predictions == 0).sum()
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    📂 Total Records
                </div>
                <div class="card-value">
                    {len(df)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    🧪 Test Records
                </div>
                <div class="card-value">
                    {len(X_test)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    🎯 Accuracy
                </div>
                <div class="card-value">
                    {best_accuracy:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">
                    🚨 Attacks Detected
                </div>
                <div class="card-value">
                    {predicted_attacks}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    st.markdown(
        "## 🤖 Model Comparison"
    )


    comparison_df = pd.DataFrame({

        "Algorithm":
            list(results.keys()),

        "Accuracy (%)":
            [
                round(value, 2)
                for value in results.values()
            ]
    })


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        f"🏆 Model used for prediction: "
        f"{best_model_name} "
        f"({best_accuracy:.2f}% accuracy)"
    )


    st.markdown(
        "## 🚨 Security Status"
    )


    if predicted_attacks > 0:

        st.warning(
            f"⚠️ {predicted_attacks} "
            f"suspicious activities detected."
        )

    else:

        st.success(
            "✅ No suspicious activities detected."
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif st.session_state.page == "Dataset Analysis":

    st.markdown(
        "## 📂 Healthcare Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True,
        height=500
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Records",
            len(df)
        )


    with c2:

        st.metric(
            "Features",
            len(feature_columns)
        )


    with c3:

        st.metric(
            "Target",
            "Label"
        )


    st.markdown(
        "## 🛡️ Label Distribution"
    )


    label_counts = y.map({
        0: "Normal",
        1: "Attack"
    }).value_counts()


    st.bar_chart(
        label_counts
    )


# ============================================================
# AI MODELS
# ============================================================

elif st.session_state.page == "AI Models":

    st.markdown(
        "## 🤖 AI Models"
    )


    st.markdown("""
### 🌲 Random Forest

Uses multiple decision trees to classify
healthcare access activity.

### 📈 Logistic Regression

Classifies the activity using relationships
between the input features.

### 🧠 Support Vector Machine

Separates normal and suspicious activities
using a classification boundary.
""")


    model_df = pd.DataFrame({

        "Model":
            list(results.keys()),

        "Accuracy (%)":
            [
                round(x, 2)
                for x in results.values()
            ]
    })


    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# INTRUSION DETECTION
# ============================================================

elif st.session_state.page == "Intrusion Detection":

    st.markdown(
        "## 🚨 Intrusion Prediction"
    )


    display_columns = [

        col for col in [

            "Patient_ID",
            "Doctor_ID",
            "User_Role",
            "Department",
            "Access_Type",
            "Login_Attempts",
            "Access_Time",
            "Device_Type",
            "Location",
            "Patient_Record_Accessed",
            "Label",
            "Actual",
            "Predicted",
            "Risk_Level"

        ]

        if col in test_results.columns
    ]


    prediction_table = (
        test_results[
            display_columns
        ].copy()
    )


    if "Label" in prediction_table.columns:

        prediction_table["Label"] = (
            prediction_table["Label"]
            .map({
                0: "Normal",
                1: "Attack"
            })
        )


    prediction_table["Actual"] = (
        prediction_table["Actual"]
        .map({
            0: "Normal",
            1: "Attack"
        })
    )


    prediction_table["Predicted"] = (
        prediction_table["Predicted"]
        .map({
            0: "Normal",
            1: "Attack"
        })
    )


    st.dataframe(
        prediction_table,
        use_container_width=True,
        height=500
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif st.session_state.page == "Risk Analysis":

    st.markdown(
        "## ⚠️ Risk Analysis"
    )


    # --------------------------------------------------------
    # RISK COUNTS
    # --------------------------------------------------------

    risk_counts = (
        test_results[
            "Risk_Level"
        ].value_counts()
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Low Risk",
            int(
                risk_counts.get(
                    "Low",
                    0
                )
            )
        )


    with c2:

        st.metric(
            "Medium Risk",
            int(
                risk_counts.get(
                    "Medium",
                    0
                )
            )
        )


    with c3:

        st.metric(
            "High Risk",
            int(
                risk_counts.get(
                    "High",
                    0
                )
            )
        )


    # ========================================================
    # IMPORTANT:
    # DEVICE TYPE VS RISK GRAPH
    # ========================================================

    st.markdown(
        "## 📊 Device Type vs Risk"
    )


    if "Device_Type" not in test_results.columns:

        st.error(
            "Device_Type column is missing."
        )

    else:

        # Device Type × Risk Level
        device_risk = pd.crosstab(
            test_results["Device_Type"],
            test_results["Risk_Level"]
        )


        # Make sure all three risk columns exist
        for risk in [
            "Low",
            "Medium",
            "High"
        ]:

            if risk not in device_risk.columns:

                device_risk[risk] = 0


        # Keep exact order
        device_risk = device_risk[
            [
                "Low",
                "Medium",
                "High"
            ]
        ]


        # Device names
        devices = (
            device_risk.index
        )


        # X positions
        x = np.arange(
            len(devices)
        )


        # Width
        width = 0.25


        # Figure
        fig, ax = plt.subplots(
            figsize=(12, 6)
        )


        # --------------------------------------------
        # LOW RISK
        # --------------------------------------------

        bars_low = ax.bar(
            x - width,
            device_risk["Low"],
            width,
            label="Low Risk"
        )


        # --------------------------------------------
        # MEDIUM RISK
        # --------------------------------------------

        bars_medium = ax.bar(
            x,
            device_risk["Medium"],
            width,
            label="Medium Risk"
        )


        # --------------------------------------------
        # HIGH RISK
        # --------------------------------------------

        bars_high = ax.bar(
            x + width,
            device_risk["High"],
            width,
            label="High Risk"
        )


        # Title
        ax.set_title(
            "Device Type vs Risk",
            fontsize=18
        )


        # Axis labels
        ax.set_xlabel(
            "Device Type",
            fontsize=13
        )

        ax.set_ylabel(
            "Number of Records",
            fontsize=13
        )


        # X axis
        ax.set_xticks(x)

        ax.set_xticklabels(
            devices,
            rotation=20,
            ha="right"
        )


        # Legend
        ax.legend()


        # Grid
        ax.grid(
            axis="y",
            alpha=0.25
        )


        # --------------------------------------------
        # VALUES ABOVE BARS
        # --------------------------------------------

        for bars in [
            bars_low,
            bars_medium,
            bars_high
        ]:

            for bar in bars:

                height = bar.get_height()

                if height > 0:

                    ax.text(
                        bar.get_x()
                        + bar.get_width() / 2,
                        height,
                        str(
                            int(height)
                        ),
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )


        plt.tight_layout()


        # Display graph
        st.pyplot(
            fig,
            use_container_width=True
        )


        # ====================================================
        # TABLE UNDER GRAPH
        # ====================================================

        st.markdown(
            "### 📋 Device Risk Summary"
        )


        summary_table = (
            device_risk
            .reset_index()
        )


        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif st.session_state.page == "Model Comparison":

    st.markdown(
        "## 📊 Model Comparison"
    )


    comparison_df = pd.DataFrame({

        "Algorithm":
            list(results.keys()),

        "Accuracy (%)":
            [
                round(x, 2)
                for x in results.values()
            ]
    })


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        "### 📈 Accuracy Graph"
    )


    chart_df = comparison_df.set_index(
        "Algorithm"
    )


    st.bar_chart(
        chart_df
    )


    st.success(
        f"Selected model: "
        f"{best_model_name} "
        f"({best_accuracy:.2f}%)"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

elif st.session_state.page == "Confusion Matrix":

    st.markdown(
        "## 🔢 Confusion Matrix"
    )


    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=[0, 1]
    )


    cm_df = pd.DataFrame(

        cm,

        index=[
            "Actual Normal",
            "Actual Attack"
        ],

        columns=[
            "Predicted Normal",
            "Predicted Attack"
        ]
    )


    st.dataframe(
        cm_df,
        use_container_width=True
    )


    st.markdown(
        "### 📌 Explanation"
    )


    st.write(
        f"Normal correctly predicted: "
        f"{cm[0][0]}"
    )

    st.write(
        f"Normal predicted as Attack: "
        f"{cm[0][1]}"
    )

    st.write(
        f"Attack predicted as Normal: "
        f"{cm[1][0]}"
    )

    st.write(
        f"Attack correctly predicted: "
        f"{cm[1][1]}"
    )


# ============================================================
# PREDICTION RESULTS
# ============================================================

elif st.session_state.page == "Prediction Results":

    st.markdown(
        "## 📋 Prediction Results"
    )


    result_download = (
        test_results.copy()
    )


    result_download["Actual"] = (
        result_download["Actual"]
        .map({
            0: "Normal",
            1: "Attack"
        })
    )


    result_download["Predicted"] = (
        result_download["Predicted"]
        .map({
            0: "Normal",
            1: "Attack"
        })
    )


    st.dataframe(
        result_download,
        use_container_width=True,
        height=500
    )


    csv_data = (
        result_download
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        "📥 Download Prediction Results",
        data=csv_data,
        file_name="Healthcare_IDS_Predictions.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:#64748b;'>"
    "Healthcare AI IDS | Machine Learning Security Monitoring"
    "</p>",
    unsafe_allow_html=True
)