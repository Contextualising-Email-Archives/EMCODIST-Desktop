
import streamlit as st


st.set_page_config(
    page_title="EMCODIST",
    page_icon="👋",
)



st.write("# Welcome to EMCODIST Desktop Version! 👋")


INBOXES_DIR = st.text_input("Folder path to original mails" )
INBOXES_DIR = INBOXES_DIR.replace('\\','/')
st.session_state["shared"] = INBOXES_DIR


st.markdown(
    """
    This is a EMCODIST Desktop app and is open-source built specifically for
    projects involved with sensitive documents.

    **👈 Select a model from the left** 
"""
)

st.sidebar.success("Select a model from here.")
