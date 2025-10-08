import streamlit as st
# from pages.project_code import load_and_preprocess_data, train_and_predict_with_lag, prepare_and_plot


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

'''
market_df = pd.read_csv('NASDAQCOM.csv')
print(market_df.head())

food_df = pd.read_csv('food_expenditures.csv')
print(food_df.head())
'''

"""Data Preprocessing"""

'''
def load_and_preprocess_data():
    # Load raw CSVs
    market_df = pd.read_csv('NASDAQCOM.csv')
    food_df = pd.read_csv('food_expenditures.csv')

    # Rename food date column
    food_df = food_df.rename(columns={'observation_date': 'food_observation_date'})

    # Combine and drop missing values
    combined = pd.concat([market_df, food_df], axis=1)
    combined.dropna(inplace=True)

    X = combined[market_df.columns]
    y = combined[food_df.columns[1]]

    # Reset indices to align
    X = X.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Convert any date-like columns to ordinals
    for col in X.select_dtypes(include="object").columns:
        try:
            X[col] = pd.to_datetime(X[col])
            X[col + "_ordinal"] = X[col].map(pd.Timestamp.toordinal)
            X.drop(columns=[col], inplace=True)
        except (ValueError, TypeError):
            pass

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_preprocessed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)

    return y, X_preprocessed_df
'''


def load_and_preprocess_data():
    market_df = pd.read_csv('NASDAQCOM.csv')

    food_df = pd.read_csv('food_expenditures.csv')

    market_df = market_df.rename(columns={'observation_date': 'market_observation_date'})
    food_df = food_df.rename(columns={'observation_date': 'food_observation_date'})

    market_df['market_observation_date'] = pd.to_datetime(market_df['market_observation_date'])
    food_df['food_observation_date'] = pd.to_datetime(food_df['food_observation_date'])

    # Filter market_df to only keep rows where the day is 1 (to match food_df's monthly data)
    market_df = market_df[market_df['market_observation_date'].dt.day == 1]

    # Inner merge on exact date match
    combined = pd.merge(
        market_df,
        food_df,
        left_on='market_observation_date',
        right_on='food_observation_date',
        how='inner'
    )

    # Drop redundant date column if needed
    combined.drop(columns=['food_observation_date'], inplace=True)

    # Define features and target
    X = combined[market_df.columns]  # features from market_df
    y = combined[food_df.columns[1]]  # target variable from food_df (e.g., food price)

    # Reset index to align
    X = X.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Convert datetime columns in X to numeric (ordinal)
    for col in X.select_dtypes(include="object").columns:
        try:
            X[col] = pd.to_datetime(X[col])
            X[col + "_ordinal"] = X[col].map(pd.Timestamp.toordinal)
            X.drop(columns=[col], inplace=True)
        except (ValueError, TypeError):
            pass

    # Identify feature types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Define transformers
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_preprocessed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    X_preprocessed_df = pd.DataFrame(
        X_preprocessed.toarray() if hasattr(X_preprocessed, "toarray") else X_preprocessed,
        columns=feature_names
    )
    return y, X_preprocessed_df


"""Regression"""


def correlation_test(y, X_preprocessed_df, lag):
    lag = 3
    y_lagged = y.shift(-lag).dropna()
    X_aligned = X_preprocessed_df.loc[y_lagged.index]

    # Compute correlation for each feature with lagged target
    for col in X_aligned.columns:
        corr = X_aligned[col].corr(y_lagged)
        print(f"Correlation between {col} and food expenditure (lag={lag}): {corr:.3f}")


def train_and_predict_with_lag(y, X_preprocessed_df, lag):
    def create_lagged_series(y, lag):
        lag = 3
        y_lagged = y.shift(-lag).dropna()
        X_lagged = X_preprocessed_df.loc[y_lagged.index].reset_index(drop=True)
        y_lagged = y_lagged.reset_index(drop=True)
        return y_lagged, X_lagged

    y_lagged, X_lagged = create_lagged_series(y, lag)

    # valid_idx = y_lagged.dropna().index
    # X_lagged = X_preprocessed_df.loc[valid_idx].reset_index(drop=True)
    # y_lagged = y_lagged.loc[valid_idx].reset_index(drop=True)

    # Time-based split
    split_ratio = 0.8
    split_point = int(len(X_lagged) * split_ratio)
    X_train, X_test = X_lagged[:split_point], X_lagged[split_point:]
    y_train, y_test = y_lagged[:split_point], y_lagged[split_point:]

    # Ridge Linear Regression
    model = Ridge(alpha=19.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape * 100:.2f}%")

    return y_test, y_pred


def plot_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_residuals_vs_predicted(y_pred, residuals):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax)
    ax.axhline(0, linestyle="--", color="red")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_residuals_histogram(residuals):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.histplot(residuals, kde=True, bins=30, ax=ax)
    ax.axvline(0, linestyle="--", color="red")
    ax.set_title("Distribution of Residuals")
    ax.set_xlabel("Residual")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted_over_time(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(y_test)), y_test, label="Actual", marker='o')
    ax.plot(np.arange(len(y_pred)), y_pred, label="Predicted", marker='x')
    ax.set_title("Actual vs Predicted Over Time")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Target")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def prepare_and_plot(y_test, y_pred):
    residuals = y_test - y_pred

    figs = []
    figs.append(plot_actual_vs_predicted(y_test, y_pred))
    figs.append(plot_residuals_vs_predicted(y_pred, residuals))
    figs.append(plot_residuals_histogram(residuals))
    figs.append(plot_actual_vs_predicted_over_time(y_test, y_pred))
    return figs


st.subheader("Modeling the Wealth Effect")

st.write("This regression model uses data on the NASDAQ and food expenditures "
        "in the US to determine how closely related they are with a lag " \
        "that you can set below (in months)")


@st.cache_data
def load_data():
    return load_and_preprocess_data()


y, X_preprocessed_df = load_data()

lag = st.slider("Select lag value", min_value=1, max_value=20, value=3)

y_test, y_pred = train_and_predict_with_lag(y, X_preprocessed_df, lag)


figures = prepare_and_plot(y_test, y_pred)
for fig in figures:
    st.pyplot(fig, use_container_width=False)
