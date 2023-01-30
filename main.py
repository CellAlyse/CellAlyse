import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu

from apps.Analyse import analyse
from apps.cbc import cbc
from apps.home import home
from apps.render import render
from apps.wbc import wbc

icon = Image.open("style/favicon.ico")
st.set_page_config(
    page_title="CellAlyse",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="expanded",
)
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
##st.markdown('<style>' + open('style/style.css').read() + '</style>', unsafe_allow_html=True)
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home", "weiße Blutzellen", "Zählung", "Analyse", "Mikroskop"],
        icons=["house", "bar-chart-steps", "clipboard-data", "moisture", "box"],
        styles={
            "icon": {"color": "#a9dc76", "font-size": "20px"},
            "nav-link-selected": {"background-color": "#3d3b40"},
            "nav-link": {"font-size": "20px", "color": "#ffffff"},
        },
    )

    # create a multi-page app the apps are in the apps folder
if selected == "Zählung":
    cbc()
elif selected == "weiße Blutzellen":
    wbc()
elif selected == "Analyse":
    analyse()
elif selected == "Home":
    home()
elif selected == "Kamera":
    cam()
elif selected == "Mikroskop":
    render()
