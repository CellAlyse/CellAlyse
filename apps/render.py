import pyvista as pv
from pyvista import examples
import streamlit as st

def render():
    ## Using pythreejs as pyvista backend
    pv.set_jupyter_backend('pythreejs')


    ## Initialize pyvista reader and plotter
    reader = pv.STLReader("storage/stl/main_body.stl")
    plotter = pv.Plotter(
        border=True,
        window_size=[820, 820])
    plotter.background_color = "#161616"

    ## Read data and send to plotter
    mesh = reader.read()
    plotter.add_mesh(mesh, color="#a9dc76", specular=0.5, specular_power=10, smooth_shading=False, show_edges=False)

    ## Export to an external pythreejs
    model_html = "model.html"
    other = plotter.export_html(model_html, backend='pythreejs')

    ## Read the exported model
    with open(model_html, 'r') as file:
        model = file.read()

    ## Show in webpage
    st.components.v1.html(model, height=920, width=920)
