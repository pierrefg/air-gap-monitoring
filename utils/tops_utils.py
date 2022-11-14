import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar

import utils.data_utils as data_utils

from scipy import signal

def analyse_top_signals(infos):
    T_s = infos["sampling_freq_hz"]/infos["rotor_freq_hz"]
    T_n = T_s*infos["sampling_freq_hz"]
    df = infos['data']['df_clean']
    for k,v in infos['attrs'].items():
        top_signal = df[v['top_attr']].copy()
        peaks = signal.find_peaks(top_signal, distance=T_s-1)[0]
        peaks_signal = np.zeros(len(top_signal.index))*np.nan
        peaks_signal[peaks] = top_signal[peaks]
        peaks_signal -= top_signal[peaks].mean()
        # matching cosine to best peak
        best_phi, best_phi_score, maj_peak = None, -np.inf, None
        peaks_to_test = peaks[np.where(peaks<T_n)[0]]
        for p in peaks_to_test:
            phi = 2*np.pi*p/T_n
            cos = np.cos(2*np.pi/T_s*df['time']-phi)
            proj_score = np.sum(cos*peaks_signal)
            if proj_score > best_phi_score:
                best_phi_score = proj_score
                best_phi = phi
                maj_peak = p
        infos['attrs'][k]['top_infos'] = {
            'peaks': peaks,
            'phi': best_phi,
            'maj_peak': maj_peak
        }
    return infos


def display_top_signals(infos):
    attrs_to_plot = list(infos['attrs'].keys())
    cols = 1
    rows = 4
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        shared_xaxes=True,
        horizontal_spacing=0.01, 
        vertical_spacing=0.03,
        specs=[[{'secondary_y': True}]]*rows,
        subplot_titles=[infos['attrs'][attr]['top_attr'] for attr in attrs_to_plot]
    )

    df = infos['data']['df_clean']
    T_s = infos["sampling_freq_hz"]/infos["rotor_freq_hz"]
    for i, attr in enumerate(attrs_to_plot):
        row, col = 1+int(np.floor(i/cols)), i%cols+1
        attr_infos = infos['attrs'][attr]
        show_legend = True if i==0 else False
        top_signal = df[attr_infos['top_attr']]
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=top_signal,
                marker={
                    'color': data_utils.format_color(attr_infos['viz']['color'])
                },
                legendgroup='Top Signals',
                showlegend=show_legend,
                name='Top Signals'
            ),
            row=row, col=col
        )
        peaks = attr_infos['top_infos']['peaks']
        maj_peak = attr_infos['top_infos']['maj_peak']
        show_peaks_legend = True and show_legend
        for peak in peaks: 
            if peak == maj_peak:
                color = 'red'
                line_width = 3
            else:
                color = data_utils.format_color(attr_infos['viz']['color'])
                line_width = 1
            peak_time = df.iloc[peak]['time']
            fig.add_trace(
                go.Scatter(
                    x=[peak_time, peak_time],
                    y=[0, top_signal[peak]],
                    legendgroup = 'peaks',
                    mode='lines',
                    showlegend = False,
                    marker = {'color': color},
                    line={'width': line_width}
                ),
                row =row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=[peak_time],
                    y=[top_signal[peak]],
                    legendgroup = 'peaks',
                    mode='markers',
                    name='Detected Peaks',
                    showlegend = show_peaks_legend,
                    marker = {'color': color}
                ),
                row=row, col=col
            )
            show_peaks_legend=False
        delta=np.ptp(top_signal)
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=np.cos(2*np.pi/T_s*df['time']-attr_infos['top_infos']['phi'])/2*delta+delta/2+np.min(top_signal),
                marker={
                    'color': data_utils.format_color(attr_infos['viz']['color'])
                },
                legendgroup='cosines',
                showlegend=show_legend,
                name='Fitted Cosines'
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=df['time'],
                y=df[attr],
                marker={
                    'color': data_utils.format_color(attr_infos['viz']['color'], opacity=0.2)
                },
                legendgroup='Signals',
                showlegend=show_legend,
                name='Signals'
            ),
            secondary_y=True,
            row=row, col=col
        )


    fig.update_layout(
        xaxis_range=[df.iloc[0]['time'],df.iloc[-1]['time']],
        xaxis4_title = 'time (s)',
        margin={'l': 60, 'b': 30, 't': 30, 'r': 10}, 
        height = 850,
        xaxis4_rangeslider_visible=True,
        xaxis4_rangeslider_thickness=0.09
    )
    return fig