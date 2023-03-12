import numpy as np
import pandas as pd
import plotly.express as px
from operator import methodcaller
import os
from sklearn import linear_model
import math

def vis_angles_from_file(f):
    df = pd.read_csv(f)
    indexExpStarts = 0
    for index, row in df.iterrows():
        if row['PathIDX'] != 99:
            indexExpStarts = index
            break

    indexExpStarts = max(0, indexExpStarts-500)
    df = df.drop(range(indexExpStarts))
    df = df[df["left_right_eye_is_blinking"].str.contains("True") == False]
    df = df[df["PathIDX"].astype(int) != 99]

    df.index = np.arange(0, len(df))

    return df.loc[:,['gaze_vis_x','gaze_vis_y','target_vis_x','target_vis_y']]

def spatial_euc_error_h(df):
    # df = vis_angles_from_file(f)
    error = 0
    for index, row in df.iterrows():
        # error += abs(row['target_vis_x'] - x_regr.predict([[row['gaze_vis_x'],row['gaze_vis_y']]])[0])
        error += abs(row['target_vis_x'] - row['gaze_x_recal'])
    return error/len(df.index)

def spatial_euc_error_v(df):
    # df = vis_angles_from_file(f)
    error = 0
    for index, row in df.iterrows():
        # error += abs(row['target_vis_y'] - y_regr.predict([[row['gaze_vis_x'],row['gaze_vis_y']]])[0])
        error += abs(row['target_vis_y'] - row['gaze_y_recal'])
    return error/len(df.index)

def spatial_euc_error_c(df):
    # df = vis_angles_from_file(f)
    error = 0
    for index, row in df.iterrows():
        p = [row['target_vis_x'], row['target_vis_y']]
        # q = [x_regr.predict([[row['gaze_vis_x'],row['gaze_vis_y']]])[0], y_regr.predict([[row['gaze_vis_x'],row['gaze_vis_y']]])[0]]
        q = [row['gaze_x_recal'],row['gaze_y_recal']]
        # q = [row['gaze_vis_x'],row['gaze_vis_y']]
        error += math.dist(p, q)
    return error/len(df.index)

def spatial_euc_errors(df):
    # df = vis_angles_from_file(f)
    e_h = spatial_euc_error_h(df)
    e_v = spatial_euc_error_v(df)
    e_c = spatial_euc_error_c(df)
    return [e_c, e_h, e_v]
