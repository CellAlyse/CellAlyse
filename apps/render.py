import pyvista as pv
import streamlit as st
from stpyvista import stpyvista


def render():
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

    ## Final touches
    #plotter.view_isometric()
    plotter.background_color = 'white'

    ## Send to streamlit
    stpyvista(plotter, key="pv_cube")
