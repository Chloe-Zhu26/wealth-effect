import streamlit as st

st.set_page_config(
    page_title="Wealth Effect Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Green-themed styling
st.markdown("""
<style>
body {
    background-color: #f7fcf7;
    font-family: 'Helvetica', sans-serif;
}

.hero {
    background-color: #e6f4ea;
    padding: 2rem 2rem;
    border-left: 6px solid #2e7d32;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 2.5rem;
    color: #2e7d32;
    margin-bottom: 0.5rem;
}

.hero p {
    font-size: 1.15rem;
    color: #555;
}

.section {
    margin-top: 2rem;
    padding: 1rem 1.5rem;
    background-color: #ffffff;
    border-left: 4px solid #66bb6a;
    border-radius: 6px;
}

ul {
    padding-left: 1.5rem;
}

ul li {
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Hero section (top banner)
st.markdown("""
<div class="hero">
    <h1>Wealth Effect Analysis Dashboard</h1>
    <p>Explore how stock market trends influence consumer spending through economic modeling and visualization tools.</p>
</div>
""", unsafe_allow_html=True)

st.write("The wealth effect is a phenonemon in which increases in capital"
        " cause increases in consumers' expenditures. This can be dangerous"
        " for personal finances.")


st.markdown("""
## What's on This Site

An interactive tool exploring how stock market performance (specifically the NASDAQ index) relates to food expenditure in the U.S.


- **Literature Review**
  Summary of academic research on the wealth effect and how rising asset values can influence consumer behavior.

- **Predictive Modeling**
  A polynomial regression model that analyzes the relationship between the NASDAQ and food spending, with adjustable time lag.

- **Visualizations**
  Interactive graphs showing actual vs predicted values, residual analysis, and time series comparisons.


Explore the sections using the sidebar navigation. Adjust model parameters, inspect trends, and draw your own conclusions!
""")

# st.subheader("How This Affects You")
