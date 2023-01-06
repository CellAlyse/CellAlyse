from streamlit_option_menu import option_menu
from apps.cbc import *
from apps.wbc import *

icon = Image.open("style/favicon.ico")
st.set_page_config(
    page_title="CellAlyse",
    page_icon=icon,
    layout="centered",
    initial_sidebar_state="expanded",
)
# use option_menu instead of steamlit mutli page
def clean():
    # remove all files in tmp folder
    for file in os.listdir("storage/tmp"):
        os.remove(os.path.join("storage/tmp", file))
#
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
    selected = option_menu("Navigation", ["Home", "weiße Blutzellen", "Zählung"],
                           icons=["house", "search", "clipboard-data"],
                           styles={
                               "icon": {"color": "#a9dc76", "font-size": "20px"},
                               "nav-link-selected": {"background-color": "#3d3b40"},
                               "nav-link": {"font-size": "20px", "color": "#ffffff"},
                           })

    # create a multi-page app the apps are in the apps folder
if selected == "Zählung":
    cbc()
elif selected == "weiße Blutzellen":
    wbc()










