import ailia
import sys
import streamlit as st

# Display Python path in Streamlit app
st.write(f"Python executable being used: {sys.executable}")
st.write(f"Python version: {sys.version}")
print("Ailia module is successfully imported!")
