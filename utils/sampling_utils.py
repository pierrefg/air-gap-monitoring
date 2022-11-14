import math
import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go

import utils.data_utils as data_utils

def find_rotor_frequency(infos):
    XY_freqs = []
    df = infos['data']['df_clean']
    for attr in ['X', 'Y']:
        x = df[attr].copy().to_numpy()
        x -= np.mean(x)
        x *= signal.windows.hann(len(x))
        x = np.pad(x, 2**18-len(df.index), constant_values=0)
        f, Xmean = signal.periodogram(x, infos['sampling_freq_hz'])
        XY_freqs.append(f[np.argmax(Xmean)])
    infos['rotor_freq_hz']=np.mean(XY_freqs)
    print(f'The found rotation frequency is {infos["rotor_freq_hz"]} for a period of {infos["sampling_freq_hz"]/infos["rotor_freq_hz"]}')
    return infos

def compute_instant_angle(infos):
    T_s = infos["sampling_freq_hz"]/infos["rotor_freq_hz"]
    angular_step = 360/T_s
    df = infos['data']['df_clean']
    for attr_name, attr_value in infos['attrs'].items():
        reference_index_angle = attr_value['top_pole_degree']
        reference_index = attr_value['top_infos']['maj_peak']

        instant_angle = np.repeat(angular_step,len(df.index)).cumsum()
        instant_angle -= instant_angle[reference_index]+reference_index_angle
        min_index = reference_index-math.ceil(np.floor(np.abs(instant_angle[0])/360)*T_s)
        max_index = reference_index+math.ceil(np.floor(np.abs(instant_angle[-1])/360)*T_s)

        signal_with_angle = pd.DataFrame(
                {'signal': df[attr_name].copy(), 'rotor_angle': instant_angle}
            )
        if attr_name=='RotorCenter': signal_with_angle['theta']=df[infos['attrs']['RotorCenter']['theta_attr']].copy()
        signal_with_angle = signal_with_angle.iloc[min_index:max_index].reset_index(drop=True)
        n_rotations = math.floor(len(signal_with_angle.index)/T_s)
        print(f'{n_rotations} full rotations extracted from {attr_name}.')

        infos['attrs'][attr_name]['signal_with_angle']=signal_with_angle
    return infos

def compute_angular_centers(infos):
    hist = np.zeros(360)
    for _,v in infos['attrs'].items():
        angle = v['signal_with_angle']['rotor_angle']%360
        hist += np.histogram(angle, list(range(0,361)))[0]
    T_n = infos['sampling_freq_hz']/infos['rotor_freq_hz']
    centers = np.arange(0,360,360/T_n)[:-1]
    best_offset, best_score = 0, -np.inf
    for offset in range(0,32):
        new_centers = np.array(np.rint((centers+offset)), dtype=np.int32)%360
        score = hist[new_centers].sum()
        if score > best_score:
            best_score = score
            best_offset = offset
    infos['angular_centers'] = (centers+best_offset)%360
    return infos

def compute_synchronized_df(infos):
    T_n = infos['sampling_freq_hz']/infos['rotor_freq_hz']
    angle_bins = (infos['angular_centers']-(360/T_n/2))%360
    angle_bins = np.sort(angle_bins)
    angle_bins = np.append(angle_bins, angle_bins[0]+360)

    df_synced = pd.DataFrame()
    for attr_name, attr_value in infos['attrs'].items():
        signal_with_angle = attr_value['signal_with_angle']
        signal_to_cut = signal_with_angle['rotor_angle'].copy()%360
        signal_to_cut[signal_to_cut<angle_bins[0]]+=360
        signal_with_angle['cats'] = pd.cut(
            signal_to_cut, 
            angle_bins
        )
        groups = signal_with_angle.groupby('cats')
        df_synced[f'{attr_name}_mean'] = groups.mean()['signal']
        df_synced[f'{attr_name}_max'] = groups.max()['signal']
        df_synced[f'{attr_name}_min'] = groups.min()['signal']
        df_synced[f'{attr_name}_std'] = groups.std()['signal']
        if attr_name=='RotorCenter':
            df_synced[f'RotorTheta_mean'] = groups.mean()['theta']
            df_synced[f'RotorTheta_max'] = groups.max()['theta']
            df_synced[f'RotorTheta_min'] = groups.min()['theta']
            df_synced[f'RotorTheta_std'] = groups.std()['theta']
    df_synced['angles_centers'] = df_synced.index.categories.mid
    infos['data']['df_synced'] = df_synced
    return infos

def display_angular_data(infos):
    fig = go.Figure()
    for attr_name in ['EFaval', 'EFmédian', 'EFamont']:
        color = infos['attrs'][attr_name]['viz']['color']

        fig.add_scatterpolar(
            r = infos['attrs'][attr_name]['signal_with_angle']['signal'],
            theta = infos['attrs'][attr_name]['signal_with_angle']['rotor_angle'],
            mode='markers',
            marker={
                'color': data_utils.format_color(color),
                'size': 3
            },
            name=attr_name
        )
    rotor_swa = infos['attrs']['RotorCenter']['signal_with_angle']
    fig.add_scatterpolar(
        theta=rotor_swa['theta'],
        r=rotor_swa['signal'],
        mode='markers',
        name='Rotor Center',
        marker={
            'color': data_utils.format_color(infos['attrs']['RotorCenter']['viz']['color']),
            'size': 3
        },
    )
    fig.update_layout(margin={'l': 60, 'b': 20, 't': 30, 'r': 30})
    return fig

def display_rotor_center(infos):
    df = infos['data']['df_clean']
    fig = go.Figure().add_scatterpolar(
        theta=df['RotorTheta'],
        r=df['RotorCenter'],
        mode='markers',
        marker={
            'color': data_utils.format_color(infos['attrs']['RotorCenter']['viz']['color']),
            'size': 3
        },
    )
    fig.update_layout(margin={'l': 60, 'b': 20, 't': 30, 'r': 30})
    return fig

def display_angular_histogram(infos):
    fig = go.Figure()
    for k,v in infos['attrs'].items():
        angle = v['signal_with_angle']['rotor_angle']%360
        fig.add_trace(go.Histogram(
            x=angle,
            histnorm='percent',
            nbinsx=360,
            name=k,
            marker_color=data_utils.format_color(v['viz']['color']),
            opacity=0.75
        ))
    for center in infos['angular_centers']:
        fig.add_vline(x=center, line_color='red')
    fig.update_layout(barmode='stack')
    fig.update_layout(margin={'l': 60, 'b': 20, 't': 30, 'r': 30})
    return fig