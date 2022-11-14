import plotly.graph_objects as go

def draw_cuts(fig, infos, attr, thetas = None, add_trace_args={}):
    closed_r_cuts = infos['attrs'][attr]['r_cuts'].copy()
    # closed_r_cuts.iloc[infos['main_period']]=closed_r_cuts.loc[0] # duplicating first element to last to close the spin
    rs = closed_r_cuts.mean(axis=1).to_numpy()
    min_rs = closed_r_cuts.min(axis=1).to_numpy()
    max_rs = closed_r_cuts.max(axis=1).to_numpy()

    if thetas is None:
        closed_theta_cuts = infos['attrs'][attr]['theta_cuts'].copy()
        thetas = closed_theta_cuts.mean(axis=1)
        min_thetas = closed_theta_cuts.min(axis=1).to_numpy()
        max_thetas = closed_theta_cuts.max(axis=1).to_numpy()
        mode = 'lines+markers'
    else:
        min_thetas, max_thetas = None, None
        mode = 'lines+markers'

    if min_thetas is None:
        fig.add_trace(
            go.Scatterpolar(
                r = min_rs,
                theta = thetas,
                fill=None,
                mode='lines',
                line_color='lightgray',
                showlegend=False,
            ),
            **add_trace_args
        )

        fig.add_trace(
            go.Scatterpolar(
                r = max_rs,
                theta = thetas,
                fill='tonext',
                mode='lines',
                line_color='lightgray',
                showlegend=False,
            ),
            **add_trace_args
        )


    fig.add_trace(
        go.Scatterpolar(
            r = rs,
            theta = thetas,
            mode = mode,
            marker={
                'color': infos['attrs'][attr]['viz']['color']
            },
            name=attr,
            showlegend=True,
        ),
        **add_trace_args
    )

    if min_thetas is not None:
        for i in range(len(rs)):
            center = [rs[i], thetas[i]]
            r_min = [min_rs[i], thetas[i]]
            r_max = [max_rs[i], thetas[i]]
            theta_min = [rs[i], min_thetas[i]]
            theta_max = [rs[i], max_thetas[i]]
            for point in [r_min, r_max, theta_min, theta_max]:
                fig.add_trace(
                    go.Scatterpolar(
                        r = [center[0], point[0]],
                        theta = [center[1], point[1]],
                        mode = 'lines',
                        line={
                            'width': 2,
                            'color': 'black'
                        },
                        name=attr,
                        showlegend=False,
                    ),
                    **add_trace_args
                )
    return fig