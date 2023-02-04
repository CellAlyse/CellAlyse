from io import StringIO
from pathlib import Path
from typing import Optional, Union

import pythreejs as tjs
import pyvista as pv
import streamlit.components.v1 as components
from ipywidgets.embed import embed_minimal_html
import streamlit as st


frontend_dir = (Path(__file__).parent / "frontend").absolute()

_component_func = components.declare_component("stpyvista", path=str(frontend_dir))

def get_Meshes(renderer: tjs.Renderer) -> list[tjs.Mesh]:
    return [
        child
        for child in renderer._trait_values["scene"].children
        if isinstance(child, tjs.Mesh)
    ]


def spin_element_on_axis(
    renderer: tjs.Renderer, axis: str = "z", revolution_time: float = 4.0):
    spin_track = tjs.NumberKeyframeTrack(
        name=f".rotation[{axis}]", times=[0, revolution_time], values=[0, 6.28]
    )
    spin_clip = tjs.AnimationClip(tracks=[spin_track])

    mesh = get_Meshes(renderer)[0]
    spin_action = [tjs.AnimationAction(tjs.AnimationMixer(mesh), spin_clip, mesh)]
    return spin_action


class stpyvistaTypeError(TypeError):
    pass


class HTML_stpyvista:
    def __init__(
        self,
        plotter: pv.Plotter,
        rotation: dict = None,
        opacity_background: float = 0.0,
    ) -> None:

        model_html = StringIO()
        pv_to_tjs = plotter.to_pythreejs()

        animations = []
        if rotation:
            animations = spin_element_on_axis(pv_to_tjs, **rotation)
        else:
            pass

        pv_to_tjs._trait_values["scene"].background = None

        pv_to_tjs._alpha = True

        pv_to_tjs.clearColor = plotter.background_color.hex_rgb

        pv_to_tjs.clearOpacity = opacity_background

        embed_minimal_html(model_html, [pv_to_tjs, *animations], title="CellAlyse")
        threejs_html = model_html.getvalue()
        model_html.close()

        dimensions = plotter.window_size
        self.threejs_html = threejs_html
        self.window_dimensions = dimensions


def stpyvista(
    input: Union[pv.Plotter, HTML_stpyvista],
    horizontal_align: str = "center",
    rotation: Union[bool, dict] = None,
    opacity_background: float = 0.0,
    key: Optional[str] = None,
):
    if isinstance(input, pv.Plotter):
        input = HTML_stpyvista(
            input, rotation=rotation, opacity_background=opacity_background
        )
    elif isinstance(input, HTML_stpyvista):
        pass
    else:
        raise (stpyvistaTypeError)

    if rotation:
        has_controls = 1.0
    else:
        has_controls = 0.0

    component_value = _component_func(
        threejs_html=input.threejs_html,
        width=input.window_dimensions[0],
        height=input.window_dimensions[1],
        horizontal_align=horizontal_align,
        has_controls=has_controls,
        key=key,
        default=0,
    )

    return component_value
    
def render():
    pv.set_jupyter_backend("pythreejs")

    reader = pv.STLReader("storage/stl/main_body.stl")
    plotter = pv.Plotter(border=True, window_size=[820, 820])
    plotter.background_color = "#161616"

    mesh = reader.read()
    plotter.add_mesh(
        mesh,
        color="#a9dc76",
        specular=0.5,
        specular_power=10,
        smooth_shading=False,
        show_edges=False,
    )

    model_html = "model.html"
    other = plotter.export_html(model_html, backend="pythreejs")

    plotter.background_color = "white"

    stpyvista(plotter, key="pv_cube")
