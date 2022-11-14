# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

import utils.misc_utils as misc_utils
import utils.data_utils as data_utils
import utils.scatter_utils as scatter_utils
import utils.polar_utils as polar_utils
import utils.tops_utils as tops_utils
import utils.sampling_utils as sampling_utils

from scipy import signal
# %%
data = json.load(open('./data/infos.json'))['avg4']
infos = data_utils.generate_infos(
    csv_path=f'./data/{data["csv_file"]}', 
    amgt_name=data['name'], 
    rotor_diameter=data['rotor_diameter'],
    poles=np.array(data['poles_ccw']),
    encoche_stator_theo=data['encohe_stator'],
    attrs_infos = data['attrs']
)

df = infos['data']['df_clean']
infos['data']['df_clean']

# %%
misc_utils.create_summary(infos)

# %%
infos = sampling_utils.find_rotor_frequency(infos)

# %%
infos = tops_utils.analyse_top_signals(infos)

# %%
tops_utils.display_top_signals(infos)

# %% COMPUTING INSTANT ANGLES OF THE SIGNALS AND FILTERING FULL ROTATIONS
infos = sampling_utils.compute_instant_angle(infos)

# %%
sampling_utils.display_angular_data(infos)

# %%
sampling_utils.display_rotor_center(infos)

# %%
infos = sampling_utils.compute_angular_centers(infos)

# %%
sampling_utils.display_angular_histogram(infos)

# %%
infos = sampling_utils.compute_synchronized_df(infos)

# %%
df_synced = infos['data']['df_synced']
fig = go.Figure()
theta = data_utils.loop_signal(df_synced['angles_centers'].to_numpy())
for attr_name in ['EFaval', 'EFmédian', 'EFamont']:
    color = infos['attrs'][attr_name]['viz']['color']
    colors = [data_utils.format_color(color, opacity=op) for op in np.linspace(0.1,1,len(theta))]
    fig.add_scatterpolar(
        r = data_utils.loop_signal(df_synced[f'{attr_name}_mean'].to_numpy()),
        theta = theta,
        mode='markers+lines',
        marker={
            'color': colors,
            'size': 8
        },
        legendgroup=attr_name,
        name=attr_name
    )
    fig.add_scatterpolar(
        r = data_utils.loop_signal(df_synced[f'{attr_name}_max'].to_numpy()),
        theta = theta,
        mode='lines',
        marker={'color': data_utils.format_color(color, opacity=0.2)},
        legendgroup=attr_name,
        showlegend=False,
        name=attr_name
    )
    fig.add_scatterpolar(
        r = data_utils.loop_signal(df_synced[f'{attr_name}_min'].to_numpy()),
        theta = theta,
        mode='lines',
        fill='tonext',
        marker={'color': data_utils.format_color(color, opacity=0.2)},
        legendgroup=attr_name,
        showlegend=False,
        name=attr_name
    )
color = infos['attrs']['RotorCenter']['viz']['color']
colors = [data_utils.format_color(color, opacity=op) for op in np.linspace(0.1,1,len(theta))]
fig.add_scatterpolar(
    r = data_utils.loop_signal(df_synced['RotorCenter_mean'].to_numpy()),
    theta = data_utils.loop_signal(df_synced['RotorTheta_mean'].to_numpy()),
    mode='markers+lines',
    marker={
        'color': colors,
        'size': 8
    },
    line_color = data_utils.format_color(color),
    legendgroup='RotorCenter',
    name='RotorCenter'
)
radial_axis_mode = ['linear', 'log']
fig.update_layout(
    polar_radialaxis_type=radial_axis_mode[0],
    margin={'l': 60, 'b': 20, 't': 30, 'r': 30}
)
fig.show()

# %%
