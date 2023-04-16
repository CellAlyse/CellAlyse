from pathlib import Path
import streamlit as st

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def main():
    st.markdown(read_markdown_file("news/news.md"))