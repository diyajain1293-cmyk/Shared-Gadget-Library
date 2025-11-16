import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# =========================
# Data loading and helpers
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("Shared_Gadget_Library_Survey_Synthetic_Data.csv")
    return df


def add_target_and_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target and numeric likelihood score."""
    df = df.copy()
    positive = {
        "Definitely will use (5/5)",
        "Probably will use (4/5)",
    }
    score_map = {
        "Probably will not use (2/5)": 2,
        "Might or might not use (3/5)": 3,
        "Probably will use (4/5)": 4,
        "Definitely will use (5/5)": 5,
    }
    df["target_willing"] = df["Q39_Likelihood_to_Use"].apply(
        lambda x: 1 if x in positive else 0
    )
    df["Likelihood_Score"] = df["Q39_Likelihood_to_Use"].map(score_map)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """Return X, y and column lists for modelling."""
    X = df.drop(
        columns=[
            "Q39_Likelihood_to_Use",
            "target_willing",
            "Response_ID",
            "Timestamp",
            "Likelihood_Score",
        ]
    )
    y = df["target_willing"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    return X, y, categorical_cols, numeric_cols


# =========================
# Model training
# =========================

def train_and_evaluate_models(df: pd.DataFrame):
    """Train Decision Tree, Random Forest and Gradient Boosting models.

    Returns:
        metrics_df: train & test metrics for each model
        cv_df: cross-validation metrics
        confusion_figs: dict of model -> confusion matrix figure
        roc_combined_fig: combined ROC curve figure
        best_pipeline: best performing model pipeline
        feature_columns: list of feature columns used for training
    """
    df_mod = add_target_and_scores(df)
    X, y, categorical_cols, numeric_cols = get_feature_matrix(df_mod)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics_rows = []
    cv_rows = []
    confusion_figs = {}
    roc_curves = {}
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        # Cross-validation on training set
        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_accuracy_mean": np.mean(cv_results["test_accuracy"]),
                "cv_precision_mean": np.mean(cv_results["test_precision"]),
                "cv_recall_mean": np.mean(cv_results["test_recall"]),
                "cv_f1_mean": np.mean(cv_results["test_f1"]),
                "cv_roc_auc_mean": np.mean(cv_results["test_roc_auc"]),
            }
        )

        # Fit on full training data
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        y_train_proba = pipe.predict_proba(X_train)[:, 1]
        y_test_proba = pipe.predict_proba(X_test)[:, 1]

        metrics_rows.append(
            {
                "model": name,
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred),
                "train_recall": recall_score(y_train, y_train_pred),
                "train_f1": f1_score(y_train, y_train_pred),
                "train_roc_auc": roc_auc_score(y_train, y_train_proba),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred),
                "test_recall": recall_score(y_test, y_test_pred),
                "test_f1": f1_score(y_test, y_test_pred),
                "test_roc_auc": roc_auc_score(y_test, y_test_proba),
            }
        )

        # Confusion matrix figure (black & white)
        cm = confusion_matrix(y_test, y_test_pred)
        fig_cm, ax_cm = plt.subplots()
        ax_cm.imshow(cm, cmap="Greys")
        ax_cm.set_title(f"Confusion Matrix - {name}")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Not willing", "Willing"])
        ax_cm.set_yticklabels(["Not willing", "Willing"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center", color="black")

        plt.tight_layout()
        confusion_figs[name] = fig_cm

        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_auc = roc_auc_score(y_test, y_test_proba)
        roc_curves[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

    # Combined ROC figure
    fig_roc, ax_roc = plt.subplots()
    for name, roc_data in roc_curves.items():
        ax_roc.plot(roc_data["fpr"], roc_data["tpr"], label=f"{name} (AUC = {roc_data['auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curves - All Models")
    ax_roc.legend(loc="lower right")
    plt.tight_layout()
    roc_combined_fig = fig_roc

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    cv_df = pd.DataFrame(cv_rows).set_index("model")

    # Choose best model by test ROC-AUC
    best_name = metrics_df["test_roc_auc"].idxmax()
    best_pipeline = pipelines[best_name]

    feature_columns = X.columns.tolist()

    return metrics_df, cv_df, confusion_figs, roc_combined_fig, best_pipeline, feature_columns


# =========================
# Filtering for insights
# =========================

def apply_filters(df: pd.DataFrame):
    df = add_target_and_scores(df)

    st.sidebar.header("Filters")

    # Equipment columns (borrow/interest)
    equipment_cols = [col for col in df.columns if col.startswith("Q15_") or col.startswith("Q19_")]
    equipment_nice = {col: col.replace("Q15_", "").replace("Q19_", "") for col in equipment_cols}

    selected_labels = st.sidebar.multiselect(
        "Equipment people are willing to borrow/lend",
        options=list(equipment_nice.values()),
        default=list(equipment_nice.values())[:5],
    )
    selected_cols = [col for col, nice in equipment_nice.items() if nice in selected_labels]

    min_like = st.sidebar.slider(
        "Minimum likelihood / satisfaction score",
        min_value=int(df["Likelihood_Score"].min()),
        max_value=int(df["Likelihood_Score"].max()),
        value=int(df["Likelihood_Score"].median()),
    )

    df_filtered = df[df["Likelihood_Score"] >= min_like].copy()

    if selected_cols:
        mask_equipment = (df_filtered[selected_cols] == 1).any(axis=1)
        df_filtered = df_filtered[mask_equipment]

    return df_filtered, equipment_cols


# =========================
# Charts for insights
# =========================

def show_insight_charts(df_filtered: pd.DataFrame, equipment_cols):
    st.subheader("1. Age group vs willingness to use BorrowBox")
    age_pref = (
        df_filtered.groupby("Q1_Age_Group")["target_willing"]
        .mean()
        .reset_index(name="share_willing")
        .sort_values("Q1_Age_Group")
    )
    chart_age = (
        alt.Chart(age_pref)
        .mark_bar()
        .encode(
            x=alt.X("Q1_Age_Group:N", title="Age group"),
            y=alt.Y("share_willing:Q", title="Share willing to use"),
            tooltip=["Q1_Age_Group", alt.Tooltip("share_willing", format=".2f")],
        )
    )
    st.altair_chart(chart_age, use_container_width=True)

    st.subheader("2. Subscription willingness by age group (monthly fee)")
    wtp_age = (
        df_filtered.groupby(["Q1_Age_Group", "Q30_WTP_Monthly"])
        .size()
        .reset_index(name="count")
    )
    chart_wtp_age = (
        alt.Chart(wtp_age)
        .mark_bar()
        .encode(
            x=alt.X("Q1_Age_Group:N", title="Age group"),
            y=alt.Y("count:Q", stack="normalize", title="Proportion"),
            color=alt.Color("Q30_WTP_Monthly:N", title="Monthly subscription willingness"),
            tooltip=["Q1_Age_Group", "Q30_WTP_Monthly", "count"],
        )
    )
    st.altair_chart(chart_wtp_age, use_container_width=True)

    st.subheader("3. Most popular equipment categories")
    eq_interest = (
        df_filtered[equipment_cols]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "equipment", 0: "interest_rate"})
    )
    eq_interest["equipment"] = eq_interest["equipment"].str.replace("Q15_", "", regex=False).str.replace("Q19_", "", regex=False)

    chart_eq = (
        alt.Chart(eq_interest.head(15))
        .mark_bar()
        .encode(
            x=alt.X("interest_rate:Q", title="Share of respondents interested"),
            y=alt.Y("equipment:N", sort="-x", title="Equipment"),
            tooltip=["equipment", alt.Tooltip("interest_rate", format=".2f")],
        )
    )
    st.altair_chart(chart_eq, use_container_width=True)

    st.subheader("4. Income vs likelihood to use (heatmap)")
    heat = (
        df_filtered.groupby(["Q4_Monthly_Income", "Likelihood_Score"])
        .size()
        .reset_index(name="count")
    )
    chart_heat = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("Q4_Monthly_Income:N", title="Monthly income"),
            y=alt.Y("Likelihood_Score:O", title="Likelihood score"),
            color=alt.Color("count:Q", title="Number of respondents"),
            tooltip=["Q4_Monthly_Income", "Likelihood_Score", "count"],
        )
    )
    st.altair_chart(chart_heat, use_container_width=True)

    st.subheader("5. Need frequency vs likelihood (boxplot)")
    box_data = df_filtered[["Q11_Need_Frequency", "Likelihood_Score"]].dropna()
    chart_box = (
        alt.Chart(box_data)
        .mark_boxplot()
        .encode(
            x=alt.X("Q11_Need_Frequency:N", title="How often do you need rarely used gadgets?"),
            y=alt.Y("Likelihood_Score:Q", title="Likelihood / satisfaction score"),
            tooltip=["Q11_Need_Frequency"],
        )
    )
    st.altair_chart(chart_box, use_container_width=True)


# =========================
# Prediction on new data
# =========================

def predict_on_new_data(best_pipeline, feature_columns):
    st.subheader("Upload dataset to predict willingness")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file with the same structure as the original survey to see predictions.")
        return

    new_df = pd.read_csv(uploaded)

    missing = [c for c in feature_columns if c not in new_df.columns]
    if missing:
        st.error("The uploaded file is missing some required columns: " + ", ".join(missing))
        return

    X_new = new_df[feature_columns]
    preds = best_pipeline.predict(X_new)
    probs = best_pipeline.predict_proba(X_new)[:, 1]

    result_df = new_df.copy()
    result_df["Predicted_Willing"] = preds
    result_df["Predicted_Willing_Prob"] = probs

    st.write("Preview of predictions:")
    st.dataframe(result_df.head())

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="BorrowBox_willingness_predictions.csv",
        mime="text/csv",
    )


# =========================
# Main app
# =========================

def main():
    st.title("BorrowBox Shared Gadget Library – Analytics & Prediction Dashboard")

    st.markdown(
        """
        This dashboard explores the **Shared Gadget Library / BorrowBox** survey data.

        - Use the **filters on the left** to focus on specific equipment and satisfaction levels.
        - Go to **Model Performance** to run the three machine learning models (Decision Tree, Random Forest, Gradient Boosting).
        - Use **Predict on New Data** to upload a fresh survey dataset and get predicted willingness labels.
        - The insights also show **which age groups prefer the service** and **how much subscription fee they are willing to pay**.
        """
    )

    df_raw = load_data()

    tab1, tab2, tab3 = st.tabs(
        [
            "Market Insights",
            "Model Performance",
            "Predict on New Data",
        ]
    )

    # ----------------------
    # Tab 1 – Insights
    # ----------------------
    with tab1:
        df_filtered, equipment_cols = apply_filters(df_raw)
        st.write(f"Filtered sample size: **{len(df_filtered)}** respondents")
        show_insight_charts(df_filtered, equipment_cols)

    # ----------------------
    # Tab 2 – Model training
    # ----------------------
    with tab2:
        st.subheader("Run classification models")
        st.markdown(
            "Click the button below to train **Decision Tree**, **Random Forest**, and **Gradient Boosting** models "
            "to predict who is willing to use BorrowBox."
        )

        if st.button("Run models"):
            metrics_df, cv_df, confusion_figs, roc_fig, best_pipeline, feature_columns = train_and_evaluate_models(
                df_raw
            )
            st.session_state["metrics_df"] = metrics_df
            st.session_state["cv_df"] = cv_df
            st.session_state["confusion_figs"] = confusion_figs
            st.session_state["roc_fig"] = roc_fig
            st.session_state["best_pipeline"] = best_pipeline
            st.session_state["feature_columns"] = feature_columns

        if "metrics_df" in st.session_state:
            st.markdown("### Test set performance")
            st.dataframe(st.session_state["metrics_df"].style.format("{:.3f}"))

            st.markdown("### 5-fold cross-validation performance")
            st.dataframe(st.session_state["cv_df"].style.format("{:.3f}"))

            st.markdown("### Confusion matrices (black & white)")
            for name, fig in st.session_state["confusion_figs"].items():
                st.write(name)
                st.pyplot(fig)

            st.markdown("### ROC curves for all models")
            st.pyplot(st.session_state["roc_fig"])
        else:
            st.info("Models have not been run yet. Click **Run models** to see performance.")

    # ----------------------
    # Tab 3 – New data prediction
    # ----------------------
    with tab3:
        if "best_pipeline" not in st.session_state or "feature_columns" not in st.session_state:
            st.info("Please run the models in the **Model Performance** tab first so we can use the best model.")
        else:
            predict_on_new_data(
                st.session_state["best_pipeline"],
                st.session_state["feature_columns"],
            )


if __name__ == "__main__":
    main()
