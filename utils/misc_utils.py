import plotly.graph_objects as go
import numpy as np
import utils.data_utils as data_utils

def create_summary(infos):
    fig = go.Figure()
    thetas = infos['poles']['degrees']
    text_positions = []
    for theta in thetas:
        vertical = 'bottom' if 0<theta<180 else 'top'
        if theta in [0, 180]: vertical='middle'
        horizontal = 'right' if 90<theta<270 else 'left'
        if theta in [90, 270]: horizontal='center'
        text_positions.append(f'{vertical} {horizontal}')

    radius = infos['rotor_diameter']/2

    # Display rotor poles as dots
    fig.add_trace(
        go.Scatterpolar(
            r = np.ones(len(thetas))*radius,
            theta = thetas,
            text=list(infos['poles']['numbers']),
            textposition=text_positions,
            fill=None,
            mode='markers+text',
            line_color='black',
            showlegend=False,
        )
    )

    top_entrefer_color = infos['attrs']['EFamont']['viz']['color']
    fig.add_trace(
        go.Scatterpolar(
            r = [0, radius],
            theta = [0, infos['attrs']['EFamont']['top_pole_degree']],
            mode='lines',
            line_color=data_utils.format_color(top_entrefer_color),
            showlegend=True,
            name='Top Entrefer Position'
        )
    )

    top_stator_color = infos['attrs']['RotorCenter']['viz']['color']
    fig.add_trace(
        go.Scatterpolar(
            r = [0, radius],
            theta = [0, infos['attrs']['RotorCenter']['top_pole_degree']],
            mode='lines',
            line_color=data_utils.format_color(top_stator_color),
            showlegend=True,
            name='Top Stator Position'
        )
    )

    fig.update_layout(
        margin={'l': 60, 'b': 50, 't': 50, 'r': 30}, 
        title=f'{infos["name"]}: placement des {len(thetas)} poles et des tops',
        width=700,
        height=600,
    )
    return fig