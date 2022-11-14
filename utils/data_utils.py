import numpy as np
import pandas as pd

def format_color(rgb_arr, opacity=None):
    if opacity is None:
        return f'rgb({str(rgb_arr)[1:-1]})'
    else:
        return f'rgba({str(rgb_arr)[1:-1]},{opacity})'

def loop_signal(np_arr):
    return np.append(np_arr, np_arr[0])

def trim_nans(pdf):
    df = pdf.copy()
    idx = df.fillna(method='ffill').dropna().index
    res_idx = df.loc[idx].fillna(method='bfill').dropna().index
    df = df.loc[res_idx] # removing null values at the end and at the begining
    df = df.reset_index(drop=True) # reset_index
    return df

def gen_attr_infos(
    top_attr, 
    theta_attr, 
    color, 
    axis, 
    top_pole_degree
):
    return {
        'top_attr': top_attr,
        'theta_attr': theta_attr,
        'top_pole_degree': top_pole_degree,
        'top_infos' : {
            'peaks': None,
            'phi': None,
            'maj_peak': None
        },
        'viz': {
            'color': color,
            'yaxis': axis
        }
    }

def generate_infos(
    csv_path, 
    amgt_name, 
    rotor_diameter,
    poles,
    encoche_stator_theo,
    attrs_infos,
    sampling_freq_hz=50
):
    # Reading and triming data
    df = pd.read_csv(csv_path)
    dfc = trim_nans(df.dropna(axis='columns', how ='all'))
    dfc['X'] /= 1000
    dfc['Y'] /= 1000
    
    # Converting rotor center from Cartesian to Polar
    Xrotor, Yrotor = attrs_infos['rotor_center']['Xattr_name'], attrs_infos['rotor_center']['Yattr_name']
    dfc['RotorCenter'] = np.sqrt(np.power(dfc[Xrotor],2)+np.power(dfc[Yrotor],2))
    pi_add = np.zeros_like(dfc[Xrotor], dtype=np.float64)
    pi_add[dfc[Xrotor]<0] = np.pi
    dfc['RotorTheta'] = (np.arctan(dfc[Yrotor]/dfc[Xrotor])+pi_add)/np.pi*180
    time = np.repeat(1/50, len(dfc.index))
    time[0] = 0
    dfc['time'] = time.cumsum()
    # Creating infos holder
    encoche_stator_degree = np.abs(poles-encoche_stator_theo).argmin()*(360/len(poles))
    infos = {
        'name': amgt_name,
        'rotor_diameter': rotor_diameter,
        'data': {
            'df_original': df,
            'df_clean': None,
            'df_sync': None,
        },
        'sampling_freq_hz': sampling_freq_hz,
        'rotor_freq_hz': None,
        'angular_centers': None,
        'poles': {
            'numbers': poles,
            'degrees': list(np.linspace(0, 360-360/len(poles), len(poles)))
        },
        'attrs': {
            attrs_infos['entrefer_am']['attr_name']: gen_attr_infos(
                attrs_infos['entrefer_am']['top_tour'], 
                'EFTheta', [2,166,118], 'y', 0
            ),
            attrs_infos['entrefer_md']['attr_name']: gen_attr_infos(
                attrs_infos['entrefer_md']['top_tour'],
                'EFTheta', [240,148,31], 'y', 0
            ),
            attrs_infos['entrefer_av']['attr_name']: gen_attr_infos(
                attrs_infos['entrefer_av']['top_tour'],
                'EFTheta', [122,87,122], 'y', 0
            ),
            'RotorCenter': gen_attr_infos(
                'TopStator', 
                'RotorTheta', [189,42,46], 'y2', encoche_stator_degree
            ),
        }
    }

    for _, attr_value in infos['attrs'].items():
        dfc[attr_value['top_attr']] -= dfc[attr_value['top_attr']].median()

    infos['data']['df_clean'] = dfc

    return infos