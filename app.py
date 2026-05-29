import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

st.title("SMOTE Demo")

st.write(
    "This app demonstrates how SMOTE creates synthetic examples "
    "to balance an imbalanced classification dataset."
)

dataset_option = st.selectbox(
    "Choose dataset",
    ["Synthetic imbalanced dataset", "Breast cancer dataset"]
)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

if dataset_option == "Synthetic imbalanced dataset":
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        weights=[0.90, 0.10],
        random_state=100
    )

    df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    df["class"] = y

else:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["class"] = data.target


target_col = "class"

X = df.drop(columns=[target_col])
y = df[target_col]


# ---------------------------------------------------------
# Function to plot data
# ---------------------------------------------------------

def plot_data(X_data, y_data, title):
    """
    Creates a 2D scatter plot.
    If the dataset has more than 2 features, PCA is used to reduce it to 2 dimensions.
    """

    if isinstance(X_data, pd.DataFrame):
        X_values = X_data.values
    else:
        X_values = X_data

    if X_values.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_plot = pca.fit_transform(X_values)
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"
    else:
        X_plot = X_values
        xlabel = "Feature 1"
        ylabel = "Feature 2"

    y_series = pd.Series(y_data)

    fig, ax = plt.subplots(figsize=(7, 5))

    for class_value in sorted(y_series.unique()):
        idx = y_series == class_value
        ax.scatter(
            X_plot[idx, 0],
            X_plot[idx, 1],
            label=f"Class {class_value}",
            alpha=0.7
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# ---------------------------------------------------------
# Display original dataset
# ---------------------------------------------------------

st.subheader("Original dataset")
st.dataframe(df)

st.subheader("Class distribution before SMOTE")
st.write(Counter(y))

st.subheader("Data plot before SMOTE")
fig_before = plot_data(X, y, "Before SMOTE")
st.pyplot(fig_before)


# ---------------------------------------------------------
# Apply SMOTE
# ---------------------------------------------------------

if st.button("Apply SMOTE"):

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_col] = y_resampled

    st.subheader("Class distribution after SMOTE")
    st.write(Counter(y_resampled))

    st.subheader("Data plot after SMOTE")
    fig_after = plot_data(X_resampled, y_resampled, "After SMOTE")
    st.pyplot(fig_after)

    st.subheader("Resampled dataset")
    st.dataframe(df_resampled)

    csv = df_resampled.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download resampled CSV",
        csv,
        "smote_resampled_dataset.csv",
        "text/csv"
    )
