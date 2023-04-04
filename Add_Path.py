import streamlit as st

INBOXES_DIR = st.text_input("Enter the directory path where the original mails can be found" )
INBOXES_DIR = INBOXES_DIR.replace('\\','/')
'''
if 'INBOXES_DIR' not in st.session_state:
    st.session_state.INBOXES_DIR = INBOXES_DIR
    '''

with st.sidebar:
    st['INBOXES_DIR'] = INBOXES_DIR