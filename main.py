# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import utils.sampling_utils as sampling_utils

import utils.misc_utils as misc_utils
import utils.data_utils as data_utils
import utils.scatter_utils as scatter_utils
import utils.polar_utils as polar_utils
import utils.tops_utils as tops_utils


# %% Organizing data infos 
# All data is in polar coordinates organized counterclockwise and all lenghths are mm and angles are in degrees
data = json.load(open('./data/infos.json'))['avg4']
infos = data_utils.generate_infos(
    csv_path=f'./data/{data["csv_file"]}', 
    amgt_name=data['name'], 
    rotor_diameter=data['rotor_diameter'],
    poles=np.array(data['poles_ccw']),
    encoche_stator_theo=data['encohe_stator'],
    attrs_infos = data['attrs']
)
infos['data']['df_clean']

# %%
misc_utils.create_summary(infos)

# %%
infos = sampling_utils.find_rotor_frequency(infos)

# %%
infos = tops_utils.analyse_top_signals(infos)

# %%
tops_utils.display_top_signals(infos)


# %%
infos = data_utils.find_shifts(infos)

# %%
fig = go.Figure()

for attr in list(infos['attrs'].keys()):
    scatter_utils.add_scatter(fig, infos, attr)

scatter_utils.update_scatter_layout(fig, yaxes=[{'name': 'Entrefer'}, {'name': 'Position Rotor'}], args={'height': 650})
fig.show()

# %% Finding the shift of each attribute
reference_attr = infos['attrs']['EFamont']
for key, value in infos['attrs'].items():
    shift = data_utils.find_shift(value, reference_attr, infos['main_period'])
    infos['attrs'][key]['shift'] = shift
    print(f'{key} has shift {shift}')

# %% Syncing data to correct the shift and adding entrefer degrees
df_synced = df_filtered.copy()
for key, value in infos['attrs'].items():
    top_attr = value['top_attr']
    shift = value['shift']
    df_synced[key] = df_filtered[key].shift(shift)
    df_synced[top_attr] = df_filtered[top_attr].shift(shift)
tops = df_synced[df_synced[list(top_attrs)].sum(axis=1)==3].index
df_synced = df_synced.loc[tops[0]:tops[-1]]

# %%
fig = go.Figure()

for key, value in infos['attrs'].items():
    scatter_utils.add_scatter(fig, df_synced, key, value)

scatter_utils.update_scatter_layout(fig, yaxes=[{'name': 'Entrefer'}, {'name': 'Position Rotor'}], args={'height': 650})
fig.show()

# %% Cutting the radius and thetas of the full dataset into on chunk per spin
for key, value in infos['attrs'].items():
    # print(value1)
    series_tops = value['tops']
    r_series = df_filtered[key]
    theta_series= df_filtered[value['theta_attr']]
    r_cuts = pd.DataFrame()
    theta_cuts = pd.DataFrame()
    for i in range(len(series_tops)-1):
        spin_id_begin, spin_id_end = series_tops[i], series_tops[i+1]-1
        spin_length = spin_id_end-spin_id_begin+1
        if spin_length!=infos['main_period']:
            print(f'Détection d\'un tour de longeur {spin_length} pour une période détectée de {infos["main_period"]} sur attribut {key}. Essaie de découpage...')
            if spin_length%infos['main_period']==0:
                print(f'Découpage en {int(spin_length/infos["main_period"])} sous tours!')
                for j in range(int(spin_length/infos['main_period'])):
                    sub_spin_id_begin=spin_id_begin
                    sub_spin_id_end=spin_id_begin+infos['main_period']-1
                    r_cuts[f'Tour {i}']=r_series.loc[sub_spin_id_begin:sub_spin_id_end].to_numpy()
                    theta_cuts[f'Tour {i}']=theta_series.loc[sub_spin_id_begin:sub_spin_id_end].to_numpy()
                    spin_id_begin = sub_spin_id_end+1
        else:
            r_cuts[f'Tour {i}']=r_series.loc[spin_id_begin:spin_id_end].to_numpy()
            theta_cuts[f'Tour {i}']=theta_series.loc[spin_id_begin:spin_id_end].to_numpy()
    infos['attrs'][key]['r_cuts']=r_cuts
    infos['attrs'][key]['theta_cuts']=theta_cuts

# %%
fig = go.Figure()

polar_utils.draw_cuts(fig, infos, 'RotorCenter')

fig.update_layout(
    margin={'l': 60, 'b': 50, 't': 50, 'r': 30}, 
)

fig.show()


# %%
attrs_to_plot = ['EFamont', 'EFmédian', 'EFaval']
fig = make_subplots(
    rows=1, 
    cols=3,
    specs=[
        [{'type': 'polar'},
        {'type': 'polar'},
        {'type': 'polar'}]
    ],
    subplot_titles=attrs_to_plot
)

for i, attr in enumerate(attrs_to_plot):
    polar_utils.draw_cuts(fig, infos, attr, add_trace_args={'col':i+1, 'row': 1}, thetas=infos['poles']['degrees'])
    # utils.draw_cuts(fig, infos, 'RotorCenter', add_trace_args={'col':i+1, 'row': 1})

# fig.update_layout(title_pad_l=0.5)
fig.update_layout(
    margin={'l': 60, 'b': 50, 't': 50, 'r': 30}, 
)

fig.show()

# %%
fig = go.Figure()

for attr in ['EFamont', 'EFmédian', 'EFaval']:
    polar_utils.draw_cuts(fig, infos, attr, thetas=infos['poles']['degrees'])

fig.update_layout(
    margin={'l': 60, 'b': 50, 't': 50, 'r': 30}, 
)

fig.show()

# %%



