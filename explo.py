# %%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

import utils.misc_utils as misc_utils
import utils.data_utils as data_utils
import utils.scatter_utils as scatter_utils
import utils.polar_utils as polar_utils
import utils.tops_utils as tQops_utils

from scipy import signal

# %% Organizing data infos 
# All data is in polar coordinates organized counterclockwise and all lenghths are mm and angles are in degrees
data = json.load(open('./data/infos.json'))['prg1']
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
df['sin32']=np.sin(np.array(df.index, dtype=np.float64)/50*2*np.pi*(50/32))
for attr in ['sin32', 'EFaval', 'EFmédian', 'EFamont', 'RotorCenter']:
    x = df[attr].copy().to_numpy()
    x -= np.mean(x)
    x *= signal.windows.blackman(len(x))
    x = np.pad(x, 2**18-len(df.index), constant_values=0)
    f, Xmean = signal.periodogram(x, 50)
    print(f'{attr}: {50/f[np.argmax(Xmean)]}')

# %%
results = {}
for attr in ['EFaval', 'EFmédian', 'EFamont', 'RotorCenter']:
    x = df[attr].copy().to_numpy()
    x -= np.mean(x)
    x *= signal.windows.hann(len(x))
    x = np.pad(x, 2**18-len(df.index), constant_values=0)
    f, Xmean = signal.periodogram(x, 50)

    results[attr] = {
        'max_freq': f[np.argmax(Xmean)],
        'max_freq_magn': np.max(Xmean),
        'fft_res': [f, Xmean]
    }

freqs = [val['max_freq'] for _, val in results.items()]
min_freq, max_freq = np.min(freqs), np.max(freqs)
min_disp_freq, max_disp_freq = 0.9*min_freq, 1.1*max_freq

fig = go.Figure()
for attr, value in results.items():
    color = infos['attrs'][attr]['viz']['color']
    fig = fig.add_scatter(
        x = value['fft_res'][0],
        y = value['fft_res'][1],
        name = attr,
        marker={
            'color': color
        }
    ).add_scatter(
        x=[value['max_freq'], value['max_freq']],
        y=[0, value['max_freq_magn']],
        marker={
            'color': color
        },
        showlegend=False
    )

fig.update_xaxes(range=[min_disp_freq, max_disp_freq])
fig.update_layout(
    margin={'l': 60, 'b': 10, 't': 30, 'r': 30},
)
print(f'The average frequency is {np.mean(freqs)} (period of {50/np.mean(freqs)}).')
fig.show()

# %%
x = df['EFaval'].copy().to_numpy()
x -= np.mean(x)
go.Figure().add_scatter(
    x = f,
    y = x
)

# %%
x = df['EFaval'].copy().to_numpy()
xres = signal.resample(x, len(x)*10)
xres -= np.mean(xres)
# x *= signal.windows.hann(len(x))
xres = np.pad(xres, 2**18-len(df.index), constant_values=0)
go.Figure().add_scatter(
    x = list(range(0,len(xres))),
    y = xres
)
f, Xmean = signal.periodogram(xres, 500)
go.Figure().add_scatter(
    x = f,
    y = Xmean
)

# %% EFstatorAM, EFstatorAV
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['EFstatorAM'],
        name='EFstatorAM'
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['EFstatorAV'],
        name='EFstatorAV'
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['EFaval'],
        name='EFaval'
    )
)
fig.update_layout(
    margin={'l': 60, 'b': 10, 't': 30, 'r': 30}
)
# %%
fig = make_subplots(specs=[[{"secondary_y": True}]])
df = infos['data']['df_clean']

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['RotorCenter'],
        name='RotorCenter'
    ),
    secondary_y=False
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['ModInstDéplArbre'],
        name='ModInstDéplArbre'
    ),
    secondary_y=True
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['ModCCdéplArbre'],
        name='ModCCdéplArbre'
    ),
    secondary_y=True
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['ModMoyDéplArbre'],
        name='ModMoyDéplArbre'
    ),
    secondary_y=True
)
fig.update_layout(
    margin={'l': 60, 'b': 10, 't': 30, 'r': 30}
)

# %%
# fig = make_subplots(specs=[[{"secondary_y": True}]])
fig= go.Figure()
df = infos['data']['df_clean']
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['VIBccAM'],
        name='EFccAM'
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['EFccMD'],
        name='EFccMD'
    )
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['EFccAV'],
        name='EFccAV'
    )
)

fig.update_layout(
    margin={'l': 60, 'b': 10, 't': 30, 'r': 30}
)
# %%
fig= go.Figure()
df = infos['data']['df_clean']
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df['EFaval'],
        name='EFaval'
    )
)

fig.update_layout(
    margin={'l': 60, 'b': 10, 't': 30, 'r': 30}
)
# %%
x = df['EFaval'].to_numpy()-df['EFaval'].mean()
fmin, fmax = 50/31.8, 50/32.2
X = np.abs(signal.zoom_fft(x, [fmin, fmax], len(x), fs=50))
f = np.linspace(fmin, fmax, len(x))
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = f,
        y = X
    )
)
fig.update_layout(
    margin={'l': 60, 'b': 10, 't': 30, 'r': 30}
)
fig.add_trace(
    go.Scatter(
        x = [50/32, 50/32],
        y = [np.min(X), np.max(X)]
    )
)
most_intense_frequency = f[np.argmax(X)]
fig.add_trace(
    go.Scatter(
        x = [most_intense_frequency, most_intense_frequency],
        y = [np.min(X), np.max(X)]
    )
)
fig.show()
# %%
def fundamental_frequency_finder(px, fs, delta = 0.00000001):
    # local iterative fft algorithm
    x = px.copy()
    #x-=np.mean(px)
    f, Xnorm = signal.periodogram(x, fs)
    Xnorm[0]=-1
    index_max_freq = np.argmax(Xnorm)
    old_max_freq = f[index_max_freq]
    max_freq = old_max_freq+10*delta
    n=1
    while(np.abs(old_max_freq-max_freq)>delta and n<20):
        fmin, fmax = f[index_max_freq-2], f[index_max_freq+2]
        # print(fmin, fmax)
        Xnorm = np.abs(signal.zoom_fft(x, [fmin, fmax], len(x), fs=50))
        f = np.linspace(fmin, fmax, len(x))
        index_max_freq = np.argmax(Xnorm)
        old_max_freq = max_freq
        max_freq = f[index_max_freq]
        n+=1
    return max_freq
# signal_sum = df['EFamont']+df['EFmédian']+df['EFaval']+df['RotorCenter']
# 50*1/fundamental_frequency_finder(signal_sum.to_numpy(), fs=50)
# %%
for attr in ['EFaval', 'EFmédian', 'EFamont', 'RotorCenter', 'TopRotorAM']:
    print(f'{attr}: {50/32-fundamental_frequency_finder(df[attr].to_numpy(), 50)}')
    print(f'{attr}: {fundamental_frequency_finder(df[attr].to_numpy(), 50)}')