import streamlit as st

st.title("🇰🇪 TaxRafiki AI")
st.subheader("Your Kenyan Tax Partner")

with st.sidebar:
    st.info("I am an AI assistant trained on KRA laws.")

user_query = st.text_input("Ask me about KRA Amnesty or Turnover Tax:")
if user_query:
    st.write(f"You asked: {user_query}. (AI Logic coming in Step 2!)")

