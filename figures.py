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
from scipy import interpolate

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'

mColor = '#2F3DEB'
aColor = '#3A9F09'
cColor = '#EB5F23'
mName = '\overline{\delta_m}'
aName = '\delta_a'
cName = '\delta_c'
rcrName = 'e'
rcaName = '\\theta'

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

def correct_profile(df_synced, ef_name):
    d2 = infos['rotor_diameter']/2
    l, theta, r, beta  = df_synced[ef_name], df_synced['RotorTheta_mean'], df_synced['RotorCenter_mean'], df_synced['angles_centers']
    d2l = d2+l
    e = np.sqrt(
        r**2+d2l**2+2*r*d2l*np.cos(beta-theta)
    )-d2
    d2e = d2+e
    alpha = beta+np.arccos(
        (d2l**2+d2e**2-r**2)/(2*d2l*d2e)
    )
    return alpha, e

def compute_ec(df_synced, ef_name):
    d2 = infos['rotor_diameter']/2
    ec = []
    for index, row in df_synced.iterrows():
        min_d = np.inf
        #position of S
        r1, t1 = d2+row[f'e_{ef_name}'], row[f'a_{ef_name}']
        for index2, row2 in df_synced.iterrows():
            #position of O_r
            r2, t2 = row2['RotorCenter_mean'], row2['RotorTheta_mean']
            d = np.sqrt(r1**2+r2**2-2*r1*r2*np.cos(t2-t1))-d2
            if d<min_d: min_d=d
        ec.append(min_d)
    return ec

names={
    'EFamont_mean': 'am',
    'EFmédian_mean': 'me',
    'EFaval_mean': 'av'
}
for ef_name in names.keys():
    df_synced['a_'+names[ef_name]], df_synced['e_'+names[ef_name]] = correct_profile(df_synced, ef_name)
    df_synced['ec_'+names[ef_name]] = compute_ec(df_synced, names[ef_name])

# %%
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
theta, r = np.array(df_synced['RotorTheta_mean']), np.array(df_synced['RotorCenter_mean']*1000)
mean_theta, mean_r = stats.circmean(theta), r.mean()
r_max_index = r.argmax()
ax.scatter(
    [mean_theta],
    [mean_r],
    color='black',
    alpha=1,
    s=50,
    zorder=12,
    marker="x"
)
ax.scatter(
    [theta[r_max_index]],
    [r[r_max_index]],
    color='red',
    alpha=1,
    s=50,
    zorder=5000,
    marker="x"
)
ax.text(theta[r_max_index], r[r_max_index]+20, 'max eccentricity\n$e='+'{:.0f}\mu m$'.format(r[r_max_index]), ha='left', color='red')
ax.text(mean_theta+0.1, mean_r-25, 'Mean rotor\nposition', ha='center')
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
    s=70,
    zorder=11
)

max_r = np.max(r)
ax.set_rticks([50, 100, 150, 200, 250])
ax.set_rlabel_position(135)
ax.text(
    2.7, 0.75*max_r, f'${rcrName}$ ($\mu m$)', color='black', zorder=1000, weight="bold"
)
ax.text(
   0.3, 10, '$O_s$', color='black', zorder=1000, weight="bold"
)
ax.text(
    -0.1, 1.1*max_r, '$\\theta$ (°)', color='black', zorder=1000, weight="bold"
)
plt.savefig(f'./rotor_center.svg', bbox_inches='tight') 
# %%
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
beta, l = np.array(df_synced['angles_centers']), np.array(df_synced['EFmédian_mean'])
alpha, e, ec = np.array(df_synced['a_me']), np.array(df_synced['e_me']), np.array(df_synced['ec_me'])
min_r, max_r = l-2*np.array(df_synced['EFmédian_std']), l+2*np.array(df_synced['EFmédian_std'])
# uncorrected
ax.scatter(
    beta, l,
    color=mColor, alpha=1, s=25, zorder=12
)
ax.scatter(
    beta, l,
    color='white', alpha=1, s=60, zorder=11
)
ax.plot(
    np.append(beta, beta[0]), np.append(l, l[0]), color=mColor, lw=1, zorder=10
)
# corrected
ax.scatter(
    alpha, e,
    color=aColor, alpha=1, s=25, zorder=99
)
ax.scatter(
    alpha, e,
    color='white', alpha=1, s=60, zorder=98
)
ax.plot(
    np.append(alpha, alpha[0]),
    np.append(e, e[0]),
    color=aColor,
    lw=1,
    zorder=97
)
# with eccentricity
ax.scatter(
    alpha, ec,
    color=cColor, alpha=1, s=25, zorder=99
)
ax.scatter(
    alpha, ec,
    color='white', alpha=1, s=60, zorder=98
)
ax.plot(
    np.append(alpha, alpha[0]),
    np.append(ec, ec[0]),
    color=cColor,
    lw=1,
    zorder=97
)
plt.fill_between(
    np.append(beta, beta[0]),
    np.append(min_r, min_r[0]),
    np.append(max_r, max_r[0]),
    color = mColor,
    alpha=0.5,
    zorder=0
)
ax.set_rticks([4, 5, 6, 7])
ax.set_yticklabels(['$4mm$', '$5mm$', '$6mm$', '$7mm$'])
ax.set_rlabel_position(135)
ax.set_rlim([4,8])
ax.text(
    2.7, 0.85*8, f'${mName}[\\beta]$', color=mColor, zorder=1000, weight="bold"
)
ax.text(
    2.8, 0.74*8, f'${aName}[\\alpha]$', color=aColor, zorder=1000, weight="bold"
)
ax.text(
    2.96, 0.65*8, f'${cName}[\\alpha]$', color=cColor, zorder=1000, weight="bold"
)
ax.text(
    -0.15, 1.1*7.5, '$\\beta$', color=mColor, zorder=1000, weight="bold"
)
ax.text(
    -0.25, 1.1*7.5, '$\\alpha$', color=aColor, zorder=1000, weight="bold"
)
ax.text(
    -0.35, 1.1*7.5, '$\\alpha$', color=cColor, zorder=1000, weight="bold"
)
ax.text(
   np.pi+0.6, 4.5, '$O_s$', color='black', zorder=1000, weight="bold"
)
plt.savefig(f'./2dprofile.svg', bbox_inches='tight') 
# %%
from mpl_toolkits.mplot3d import Axes3D
e_am, e_me, e_av = np.array(df_synced['e_am']), np.array(df_synced['e_me']), np.array(df_synced['e_av'])
a_am, a_me, a_av = np.array(df_synced['a_am']), np.array(df_synced['a_me']), np.array(df_synced['a_av'])
Xam, Yam = e_am*np.cos(a_am), e_am*np.sin(a_am)
Xme, Yme = e_me*np.cos(a_me), e_me*np.sin(a_me)
Xav, Yav = e_av*np.cos(a_av), e_av*np.sin(a_av)

X,Y,Z=[],[],[]
depths = [1,2,3]
min_e = np.inf
min_e_coord = None
for i in list(range(len(Xam))):
    X.append([1,2,3])
    Y.append([a_am[i],a_me[i],a_av[i]])
    Z.append([e_am[i],e_me[i],e_av[i]])
    if e_am[i]<min_e: 
        min_e=e_am[i]
        min_e_coord=[1,a_am[i],e_am[i]]
    if e_me[i]<min_e: 
        min_e=e_me[i]
        min_e_coord=[1,a_me[i],e_me[i]]
    if e_av[i]<min_e: 
        min_e=e_av[i]
        min_e_coord=[1,a_av[i],e_av[i]]
fig = plt.figure()
ax = fig.add_subplot(projection=Axes3D.name)
ax.plot_wireframe(
    np.array(X),
    np.array(Y),
    np.array(Z),
    # facaColors=fcolors,
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
    cmap=matplotlib.cm.jet_r,
    # facaColors=fcolors,
    # rstride=20, cstride=5,
    # cstride=10,
    # color=(0,0,0,0.3),
    alpha=0.5,
    # color='black',
    # linewidth=30,
    antialiased=True,
    # shade=False
)
ax.scatter(
    min_e_coord[0],min_e_coord[1],min_e_coord[2],color='r',zorder=100000
)
ax.text(
    min_e_coord[0],min_e_coord[1]-0.4,min_e_coord[2],f'min air gap\n${aName}='+'{:.2f}'.format(min_e)+'mm$',color='r',zorder=100000
)
ax.text(1.4, 0, 9.5, f'${aName}[\\alpha]$ (mm)')
ax.set_xticks([1,2,3])
ax.set_xticklabels(['upstream', 'median', 'downstream'])
ax.set_yticks([0,2*np.pi/4,np.pi,6*np.pi/4,2*np.pi])
ax.set_yticklabels(['0°', '90°', '180°', '270°', '360°'])
ax.set_xlabel('sensor position', fontsize=10, rotation = 0)
ax.set_ylabel('$\\alpha$ (°)', fontsize=10, rotation = 0)
# plt.subplots_adjust(left=0.1, right=0.9)
# ax.set_zlabel('$e(\\alpha)$', fontsize=10, labelpad=-2)
ax.view_init(elev=10, azim=140)
# ax.
# ax.dist= 12

# plt.tight_layout()
plt.savefig(f'./3dviz.svg', bbox_inches='tight') 
# %%
swa = infos['attrs']['EFmédian']['signal_with_angle'].copy()
angle, signal = swa['rotor_angle'], swa['signal']
f = interpolate.interp1d(angle, signal, kind='linear', assume_sorted=False)
angle_min, angle_max = angle.min(), angle.max()
max_angle = 360
wanted_angles = np.linspace(0,max_angle,int(max_angle/(max_angle/32)),endpoint=True)
r = np.empty((0, len(wanted_angles)))
continuous_mean = np.empty((0, len(wanted_angles)))
steps = range(2, int(angle_max/max_angle)-2)
for i in steps:
    r = np.append(r, np.array([f(wanted_angles+i*max_angle)]), axis=0)
    continuous_mean = np.append(continuous_mean, np.array([r.mean(axis=0)]), axis=0)

fig, ax = plt.subplots()
legend='Raw $\delta_m$'
ax.plot(
    [],
    [],
    color='black',
    alpha=0.3,
    label=legend,
    zorder=999
)
for spin in r:
    ax.plot(
        wanted_angles,
        spin,
        color='black',
        alpha=0.05,
        # label=legend,
        zorder=999
    )
    legend=None
ax.plot(
    wanted_angles,
    r.mean(axis=0),
    'o-',
    color=mColor,
    alpha=1,
    # label=legend
    zorder=1000,
    label='Averaged $\overline{\delta_m}$',
    # linestyle='o-'
    ms=5
)
ax.scatter(
    wanted_angles,
    r.mean(axis=0),
    color=mColor,
    alpha=1,
    # label=legend
    zorder=1005,
    s=20
)
# ax.scatter(
#     wanted_angles,
#     r.mean(axis=0),
#     color='white',
#     alpha=0.5,
#     # label=legend
#     zorder=1001,
#     s=60
# )
ax.legend(loc="upper right")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_bounds(min(wanted_angles), max(wanted_angles))
ax.spines['left'].set_bounds(min(signal), max(signal))
ax.set(xlabel='$\\beta$ $(°)$')
ax.set(ylabel=f'Measured air gap $(mm)$')
fig.set_size_inches(8, 6, forward=True)
plt.savefig(f'./synchronous_avg.svg', bbox_inches='tight') 

# %%
