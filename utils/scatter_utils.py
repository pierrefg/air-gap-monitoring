import plotly.graph_objects as go
import numpy as np
import utils.data_utils as data_utils

def add_scatter(fig, infos, y_name):
    attr_infos = infos['attrs'][y_name]
    df_clean = infos['data']['df_clean']
    line_dict = {'color': attr_infos['viz']['color']}
    fig.add_trace(
        go.Scatter(
            mode='lines',
            x=df_clean.index,
            y=df_clean[y_name],
            name=y_name,
            line=line_dict,
            yaxis=attr_infos['viz']['yaxis']
        )
    )
    top_attr = attr_infos['top_attr']
    if top_attr is not None:
        ilocs_max = np.array(df_clean[df_clean[attr_infos['top_attr_binary']]==1].index)
        showlegend = True
        for index in ilocs_max:
            fig.add_trace(
                go.Scatter(
                    mode='lines',
                    x=[index, index],
                    y=[df_clean[y_name].min(), df_clean[y_name].max()],
                    line={
                        'color': attr_infos['viz']['color'],
                        'width': 4
                    },
                    showlegend=showlegend,
                    legendgroup=top_attr,
                    name=top_attr,
                    yaxis=attr_infos['viz']['yaxis']
                )
            )
            showlegend=False

def update_scatter_layout(fig, yaxes=[{'name': 'untitled'}], xname='record number', args={}):
    fig.update_layout(
        margin={'l': 60, 'b': 50, 't': 10, 'r': 30}, 
        xaxis_title=xname
    )
    yaxes_names = ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4']
    yaxes_args = {}
    step = 1/len(yaxes)
    curr = 0
    for i in range(len(yaxes)):
        yaxes_args[yaxes_names[i]] = dict(
            anchor="x",
            autorange=True,
            domain=[curr, curr+step],
            showline=True,
            tickmode="auto",
            zeroline=True,
            title=yaxes[i]['name']
        )
        curr+=step
    fig.update_layout(**yaxes_args)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
        ),
        **args
    )