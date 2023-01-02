import streamlit as st
from streamlit_option_menu import option_menu
from apps.cbc import *

st.set_page_config(
    page_title="CellAlyse",
    page_icon=":microscope:",
    layout="centered",
    initial_sidebar_state="expanded",
)
# use option_menu instead of steamlit mutli page
def clean():
    # remove all files in tmp folder
    for file in os.listdir("storage/tmp"):
        os.remove(os.path.join("storage/tmp", file))

hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#st.markdown('<style>' + open('style/style.css').read() + '</style>', unsafe_allow_html=True)
with st.sidebar:
    selected = option_menu("Navigation", ["Home", "weiße Blutzellen", "Zählung"],
                           icons=["house", "search", "clipboard-data"],
                           styles={
                               "icon": {"color": "#f9f871", "font-size": "20px"},
                               "nav-link-selected": {"background-color": "#807286"},
                               "nav-link": {"background-color": "#0e1117", "font-size": "19px"},
                           })

    # create a multi-page app the apps are in the apps folder
if selected == "Zählung":
    cbc()
    clean()










