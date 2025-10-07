import streamlit as st
from am_project_code import load_and_preprocess_data, train_and_predict_with_lag, prepare_and_plot

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
