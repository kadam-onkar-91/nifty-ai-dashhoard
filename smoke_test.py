# smoke_test.py -- 10-second check: is Streamlit Cloud itself serving pages?
# Deploy it as a SEPARATE app (Create app -> same repo, branch main, main file = smoke_test.py).  Your real app (app.py) is not touched.
import sys
import streamlit as st

st.set_page_config(page_title="Smoke test")
st.markdown("<style>.ok{color:#22d3ee;font-size:1.4rem}</style>", unsafe_allow_html=True)   # same kind of st.markdown+CSS as the real app
st.title("Streamlit smoke test")
st.markdown('<span class="ok">Agar ye line cyan rang me dikh rahi hai, to Streamlit Cloud + browser theek hain.</span>', unsafe_allow_html=True)
st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)
