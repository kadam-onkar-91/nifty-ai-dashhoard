import streamlit as st
import requests
import urllib.parse

def get_upstox_access_token():
    API_KEY = st.secrets["UPSTOX_API_KEY"]
    API_SECRET = st.secrets["UPSTOX_API_SECRET"]
    REDIRECT_URI = st.secrets["REDIRECT_URI"]

    st.sidebar.subheader("🔐 Upstox Live Authentication")
    
    query_params = st.query_params
    auth_code = query_params.get("code")
    
    if auth_code and not st.session_state.access_token:
        token_url = "https://api.upstox.com/v2/login/authorization/token"
        payload = {
            'code': auth_code,
            'client_id': API_KEY,
            'client_secret': API_SECRET,
            'redirect_uri': REDIRECT_URI,
            'grant_type': 'authorization_code'
        }
        headers = {'accept': 'application/json', 'Content-Type': 'application/x-www-form-urlencoded'}
        
        try:
            res = requests.post(token_url, data=payload, headers=headers)
            res_json = res.json()
            if 'access_token' in res_json:
                st.session_state.access_token = res_json['access_token']
                st.sidebar.success("Connected to Live Upstox Feed!")
            else:
                st.sidebar.error("Login failed. Check Redirect URI in Upstox Portal.")
        except Exception:
            pass

    if st.session_state.access_token:
        st.sidebar.success("🟢 Live Feed Active")
        if st.sidebar.button("Disconnect"):
            st.session_state.access_token = None
            st.rerun()
        return st.session_state.access_token
        
    encoded_redirect = urllib.parse.quote(REDIRECT_URI, safe='')
    login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={API_KEY}&redirect_uri={encoded_redirect}"
    
    # ✅ SECURE LOGIN BUTTON JISE CLICK KARNE PAR NAYA TAB KHULEGA
    st.sidebar.link_button("🚀 Login with Upstox", login_url)
    
    return None
    
