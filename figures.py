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

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

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
infos = sampling_utils.find_rotor_frequency(infos)

# %%
infos = tops_utils.analyse_top_signals(infos)

# %% COMPUTING INSTANT ANGLES OF THE SIGNALS AND FILTERING FULL ROTATIONS
infos = sampling_utils.compute_instant_angle(infos)

# %%
infos = sampling_utils.compute_angular_centers(infos)

# %%
infos = sampling_utils.compute_synchronized_df(infos)
# %%
df_synced = infos['data']['df_synced'].copy()
df_synced['RotorTheta_mean'] = 2*np.pi*(df_synced['RotorTheta_mean']/360)
df_synced['RotorTheta_std'] = 2*np.pi*(df_synced['RotorTheta_std']/360)
df_synced['angles_centers'] = 2*np.pi*(df_synced['angles_centers']/360)

# %%
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
theta, r = np.array(df_synced['RotorTheta_mean']), np.array(df_synced['RotorCenter_mean']*1000)
ax.plot(
    np.append(theta, theta[0]),
    np.append(r, r[0]),
    color='black',
    lw=1.1,
    zorder=10
)
ax.errorbar(
    theta, 
    r,
    xerr=np.array(list(zip(df_synced['RotorTheta_std'], df_synced['RotorTheta_std']))).T,
    yerr=np.array(list(zip(df_synced['RotorCenter_std']*1000*2, 2*1000*df_synced['RotorCenter_std']))).T,
    # yerr=np.array(list(zip(theta, theta))).T,
    color=(0,0,0,0),
    capsize=0, ecolor = 'red', lw=1, zorder=100
)
ax.scatter(
    theta,
    r,
    color='black',
    alpha=1,
    s=25,
    zorder=12
)
ax.scatter(
    theta,
    r,
    color='white',
    alpha=1,
    s=60,
    zorder=11
)

max_r = np.max(r)
ax.set_rticks([50, 100, 150, 200, 250])
ax.set_rlabel_position(135)
ax.text(
    2.7, 0.75*max_r, '$r$ ($\mu m$)', color='black', zorder=1000, weight="bold"
)
ax.text(
    -0.1, 1.1*max_r, '$\\theta$ (°)', color='black', zorder=1000, weight="bold"
)
# %%
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
beta, l = np.array(df_synced['angles_centers']), np.array(df_synced['EFmédian_mean'])
min_r, max_r = l-2*np.array(df_synced['EFmédian_std']), l+2*np.array(df_synced['EFmédian_std'])
ax.scatter(
    beta,
    l,
    color='black',
    alpha=1,
    s=25,
    zorder=12
)
ax.scatter(
    beta,
    l,
    color='white',
    alpha=1,
    s=60,
    zorder=11
)
ax.plot(
    np.append(beta, beta[0]),
    np.append(l, l[0]),
    color='black',
    lw=1,
    zorder=10
)
plt.fill_between(
    np.append(beta, beta[0]),
    np.append(min_r, min_r[0]),
    np.append(max_r, max_r[0]),
    color = (0,0,0,0.4),
    zorder=-30
)
ax.set_rticks([2, 4, 6])
ax.set_rlabel_position(135)

ax.text(
    2.7, 0.75*8, '$l$ ($mm$)', color='black', zorder=1000, weight="bold"
)
ax.text(
    -0.1, 1.1*7.5, '$\\beta$ (°)', color='black', zorder=1000, weight="bold"
)
# %%
lam, lme, lav = np.array(df_synced['EFamont_mean']), np.array(df_synced['EFmédian_mean']), np.array(df_synced['EFaval_mean'])
Xam, Yam = lam*np.cos(beta), lam*np.sin(beta)
Xme, Yme = lme*np.cos(beta), lme*np.sin(beta)
Xav, Yav = lav*np.cos(beta), lav*np.sin(beta)

X,Y,Z,C=[],[],[],[]
depths = [1,2,3]
for i in list(range(len(Xam)))+[0]:
    X.append([1,2,3])
    Y.append([Xam[i],Xme[i],Xav[i]])
    Z.append([Yam[i],Yme[i],Yav[i]])
    C.append([lam[i],lme[i],lav[i]])
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
scamap = plt.cm.ScalarMappable(cmap='inferno')
scamap.set_clim(5,8)
fcolors = scamap.to_rgba(np.array(C))
ax.plot_wireframe(
    np.array(X),
    np.array(Y),
    np.array(Z),
    facecolors=fcolors,
    # rstride=5, cstride=5,
    # color=(0,0,0,0.3),
    color='black',
    linewidth=1,
    antialiased=True,
    zorder=100
)
# C = np.linspace(1, 5, np.array(Z).size).reshape(np.array(Z).shape)

ax.plot_surface(
    np.array(X),
    np.array(Y),
    np.array(Z),
    # facecolors=fcolors,
    # rstride=20, cstride=5,
    # cstride=10,
    # color=(0,0,0,0.3),
    # color='black',
    # linewidth=30,
    antialiased=True,
    shade=False
)
ax.set_xticks([1,2,3])
ax.set_xticks([1,2,3])
ax.set_xticklabels(['upstream', 'median', 'downstream'])
ax.set_xlabel('sensor position', fontsize=10, rotation = 0)
ax.set_ylabel('y', fontsize=10, rotation = 0)
ax.set_zlabel('z', fontsize=10, rotation = 0)

# %%
