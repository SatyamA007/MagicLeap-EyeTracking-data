import json
import numpy as np
import pandas as pd
import plotly.express as px
from operator import methodcaller
import os
from sklearn import linear_model
import math
from scipy.stats.stats import pearsonr

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rads = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return math.degrees(rads)

def regression_models(shifted_df):
    indexExpStarts = 0
    for index, row in shifted_df.iterrows():
        if row['path'] != 99:
            indexExpStarts = index
            break

    indexExpStarts = max(0, indexExpStarts-500)
    shifted_df = shifted_df.drop(range(indexExpStarts))
    shifted_df = shifted_df[shifted_df["blinks"] == False]
    shifted_df = shifted_df[shifted_df["path"].astype(int) != 99]
    
    shifted_df.index = np.arange(0, len(shifted_df))

    X1 = shifted_df[['gaze_vis_x','gaze_vis_y']]
    y1 = shifted_df['target_vis_x']

    X2 = shifted_df[['gaze_vis_x','gaze_vis_y']]
    y2 = shifted_df['target_vis_y']
    x_regr = linear_model.LinearRegression()
    x_regr.fit(X1.values, y1.values)
    
    y_regr = linear_model.LinearRegression()
    y_regr.fit(X2.values, y2.values)

    return  [x_regr, y_regr]

def clean_df(df, callibration = False):
    indexExpStarts = 0
    for index, row in df.iterrows():
        if row['PathIDX'] != 99:
            indexExpStarts = index
            break

    indexExpStarts = max(0, indexExpStarts-200)
    df = df.drop(range(indexExpStarts))
    df["left_right_eye_is_blinking"] = df["left_right_eye_is_blinking"].apply(lambda x: True if "True" in x else False)        
    df["PathIDX"] = df["PathIDX"].apply(lambda x: 99 if (x > 0 and not callibration) else x) # Removing moving points from callibration        
    df = df.rename(columns={"left_right_eye_is_blinking": "blinks"})
    df = df.rename(columns={"PathIDX": "path"})
    df['seconds'] = df['seconds'].apply(lambda x: x-df['seconds'].iat[0])
    df['frame'] = df['frame'].apply(lambda x: x-df['frame'].iat[0])
    df = df.fillna(0)
    df.index = np.arange(0, len(df))
    return df

def shifted_euc_error_h(df, shift):
    error = 0
    count = 0
    data = df[:-shift]
    if shift == 0:
        data = df

    for index, row in data.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        error += abs(row['target_vis_x'] - df.iloc[index + shift]['gaze_vis_x'])
        error += abs(row['target_vis_y'] - df.iloc[index + shift]['gaze_vis_y'])
        count+=1
    return error/count

def shifted_df(df):
    min_error = float('inf')
    idx = 0

    for shift in range(30):
        e_after_shift = shifted_euc_error_h(df,shift)
        if min_error > e_after_shift:
            min_error = e_after_shift
            idx = shift

    if idx == 0:
        return [df, idx]

    shifted_df = df.copy(deep=True)
    shifted_df = shifted_df[:-idx]
    shifted_df['gaze_vis_x'] = df['gaze_vis_x'] 
    shifted_df['gaze_vis_y'] = df['gaze_vis_y'] 
    return [shifted_df, idx]

def spatial_euc_error_h(df):
    #Callibrated error
    error1 = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        error1 += abs(row['target_vis_x'] - row['gaze_x_recal'])
    #Callibrated error
    error2 = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        error2 += abs(row['target_vis_x'] - row['gaze_vis_x'])
    return [error1/count, error2/count]

def spatial_euc_error_v(df):
    #Callibrated error
    error1 = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        error1 += abs(row['target_vis_y'] - row['gaze_y_recal'])
    #Uncallibrated error
    error2 = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        error2 += abs(row['target_vis_y'] - row['gaze_vis_y'])
    return [error1/count, error2/count]

def spatial_euc_error_c(df):
    #Callibrated error
    error1 = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        p = [row['target_vis_x'], row['target_vis_y']]
        q = [row['gaze_x_recal'],row['gaze_y_recal']]
        error1 += math.dist(p, q)
    #Uncallibrated error
    error2 = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        p = [row['target_vis_x'], row['target_vis_y']]
        q = [row['gaze_vis_x'],row['gaze_vis_y']]
        error2 += math.dist(p, q)
    return [error1/count, error2/count]


def spatial_vec_errors(df):
    error = 0
    count = 0
    for index, row in df.iterrows():
        if row['path']==99 or row['blinks']:
            continue
        count += 1
        p = row['gaze_vector']
        q = row['target_vector']
        if isinstance(p, str):
            p = json.loads(p)
            q = json.loads(q)

        error += angle_between(p,q)
    return error/len(df.index)

def spatial_euc_errors(df):
    e_h = spatial_euc_error_h(df)
    e_v = spatial_euc_error_v(df)
    e_c = spatial_euc_error_c(df)
    return [e_c, e_h, e_v]

def pearsonr_from_df(df):
    df = df[(df.path!=99) & (df.path>0) ]
    x_g = df.gaze_vis_x.astype(float).fillna(0.0)
    x_g_recal = df.gaze_x_recal.astype(float).fillna(0.0)
    x_t = df.target_vis_x.astype(float).fillna(0.0)
    x_offset = np.subtract(x_g,x_t)
    x_offset_recal = np.subtract(x_g_recal,x_t)
    
    y_g = df.gaze_vis_y.astype(float).fillna(0.0)
    y_g_recal = df.gaze_y_recal.astype(float).fillna(0.0)
    y_t = df.target_vis_y.astype(float).fillna(0.0)
    y_offset = np.subtract(y_g,y_t)
    y_offset_recal = np.subtract(y_g_recal,y_t)
    pr = pearsonr(x_offset, y_offset)
    pr_recal = pearsonr(x_offset_recal, y_offset_recal)

    return [[pr[0], pr[1]], [pr_recal[0], pr_recal[1]]]