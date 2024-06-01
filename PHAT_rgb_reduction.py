import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.table import Table
import math
import sys
import pickle
import matplotlib
from matplotlib import rc
from scipy import optimize as opt
from scipy.ndimage.filters import gaussian_filter

path = '/mnt/data66/home/clchen/.jupyter/1M31_dust/data progress/raw_data/Star File/'
file = os.listdir(path)
files = []
a = math.radians(52)

# read data and preliminary reduction


def cal_RGB():
    for i in [file]:
        if 'hlsp' in i:
            j = i.strip('hlsp_b')
            j = j.strip('.fits')
            j = int(j)
            hduu = fits.open(path + i)
            data = Table.read(hduu)
    #         print(hduu[1].header)
            hduu.close()
    #         break
            a = list(data.columns)
            b = data.columns
            dd = pd.DataFrame()
            for l in range(77):
                tpe = b[l].dtype.name
                dd[a[l]] = np.array(b[l], dtype=tpe)
            dd = pd.DataFrame({
                'ra': data['ra'],
                'dec': data['dec'],
                'f110w': data['f110w_vega'],
                'f110w_err': data['f110w_err'],
                'f110w_snr': data['f110w_snr'],
                'f110w_sharp': data['f110w_sharp'],
                'f110w_round': data['f110w_round'],
                'f110w_crowd': data['f110w_crowd'],
                'f160w': data['f160w_vega'],
                'f160w_err': data['f160w_err'],
                'f160w_snr': data['f160w_snr'],
                'f160w_sharp': data['f160w_sharp'],
                'f160w_round': data['f160w_round'],
                'f160w_crowd': data['f160w_crowd'],
                'field': data['field']
            })
            dd['brick'] = [j] * len(dd)
            files.append(dd)
    # a,b,dd,data=[],[],[],[]
    df = pd.concat([i for i in files]).reset_index(drop=True)
    files = []

    # cal density/bin
    a = math.radians(52)
    min1 = min(df['ra'])
    max1 = max(df['ra'])
    min2 = min(df['dec'])
    max2 = max(df['dec'])

    xbin = np.arange(min1, max1+0.004166666666666667, 0.004166666666666667)
    ybin = np.arange(min2, max2+0.004166666666666667, 0.004166666666666667)
    label1 = [k for k in range(len(xbin)-1)]
    label2 = [k for k in range(len(ybin)-1)]
    cut1 = pd.cut(df['ra'], xbin, right=False, labels=label1)
    cut2 = pd.cut(df['dec'], ybin, right=False, labels=label2)

    xbin2 = np.arange(min1, max1+0.0019444444444444444, 0.0019444444444444444)
    ybin2 = np.arange(min2, max2+0.0019444444444444444, 0.0019444444444444444)
    labels1 = [k for k in range(len(xbin2)-1)]
    labels2 = [k for k in range(len(ybin2)-1)]
    cutt1 = pd.cut(df['ra'], xbin2, right=False, labels=labels1)
    cutt2 = pd.cut(df['dec'], ybin2, right=False, labels=labels2)

    df['xbin'] = cut1  # cal density
    df['ybin'] = cut2
    df['xbin2'] = cutt1  # mcmc
    df['ybin2'] = cutt2

    df['round'] = (df['f110w_round']**2)+(df['f160w_round']**2)
    df['crowd'] = (df['f110w_crowd']**2)+(df['f160w_crowd']**2)
    # u=df['f110w_sharp']*np.cos(a)+df['f160w_sharp']*np.sin(a)
    # v=df['f160w_sharp']*np.cos(a)-df['f110w_sharp']*np.sin(a)
    # c=(u/0.175)**2+(v/0.09625)**2
    sharp1 = df['f110w_sharp']
    sharp2 = df['f160w_sharp']

    # define elipse
    major_axis_length = 0.35
    position_angle = -128
    axis_ratio = 0.55

    # cal sharpness
    ellipse_equation = ((sharp1 / major_axis_length) ** 2 + (sharp2 / (major_axis_length * axis_ratio)) ** 2)
    angle_diff = np.deg2rad(sharp2 - sharp1 - position_angle)
    ellipticity = np.sqrt((sharp1 / major_axis_length) ** 2 + (sharp2 / (major_axis_length * axis_ratio)) ** 2)

    # reduction
    df['ellipse_equation'] = ellipse_equation
    df['ellipticity'] = ellipticity

    df2 = df[(df["f160w_snr"] > 5.0) & (df["f110w_snr"] > 5.0) & (df['round'] < (4.0**2)) & (df['crowd'] < (2.5**2)) &
             (df['ellipse_equation'] < 1.) & (df['ellipticity'] < 1.)]
    return (df2)


def cal_density(df2):
    excels = []
    dff = df2.groupby(by=['xbin', 'ybin'])
    s = 0
    for i in dff:
        i[1]['label'] = s
        cc = i[1][(i[1]['f110w']-i[1]['f160w'] > 0.3) & (i[1]['f160w'] > 18.5) & (i[1]['f160w'] < 21)]
        i[1]['density2'] = len(cc)
        excels.append(i[1])
        s += 1
    df3 = pd.concat([i for i in excels])
    df3.loc[df3['density2'] == 0, 'density2'] = 1
    df3['density2'] = np.log10(df3['density2']/225)
    return df3

# cut_excel_path: mag_limit_reduction
# cut_excel_path[0]:limit_F110W; cut_excel_path[1]:limit_F160W
# mag_limit


def mag_cut(df3, cut_excel_path):
    cut_excel = pd.read_excel(cut_excel_path)
    d3 = df3.sort_values(by='density2', ascending=True)
    _x110 = np.arange(np.min(cut_excel_path[0][0]), np.max(cut_excel_path[0][0]), 0.01)
    zp = np.polyfit(cut_excel_path[0][0], cut_excel_path[0][1], 10)
    zy = np.poly1d(zp)[0]
    cs = zy(d3['density2'])
    _x = np.arange(np.min(cut_excel_path[1][0]), np.max(cut_excel_path[1][0]), 0.01)
    zp2 = np.polyfit(cut_excel_path[1][0], cut_excel_path[1][1], 10)
    zy2 = np.poly1d(zp2)
    cs2 = zy2(d3['density2'])
    df4 = d3[(d3['f110w'] < cs) & (d3['f160w'] < cs2-0.5) & (d3['brick'] == 3)].reset_index(drop=True)
    return df4


# cal_unred_rgb
def fun1(x0, y0, x1, y1):
    a = y0-y1
    b = x1-x0
    c = x0*y1-x1*y0
    return a, b, c


def fun2(line0, line1):
    a0, b0, c0 = fun1(*line0)
    a1, b1, c1 = fun1(*line1)
    d0 = a0*b1-a1*b0
    if d0 == 0:
        return 0
    else:
        xx = (b0*c1-b1*c0)/d0
        yy = (a1*c0-a0*c1)/d0
        return xx, yy


def cal_unred_rgb(df4)
    d2 = df4.groupby(['xbin2', 'ybin2'])
    e = []
    for i in d2:
        j = i[1][(i[1]['f160w'] > 19.0) & (i[1]['f160w'] < 22.0)]
        j['cfiducial'] = 0.752-0.1 * (j['f160w']-22.0)-0.006*((j['f160w']-22.0)**2)
        j['offset_color'] = j['f110w']-j['f160w']-j['cfiducial']
        i[1]['cshift'] = np.mean(j['offset_color'])
        i[1]['width'] = np.std(j['offset_color'])
        e.append(i[1])
    DF = pd.concat([i for i in e]).reset_index(drop=True)
    length = []
    for i in range(len(DF)):
        b = DF['ra'][i]*1.257172+DF['dec'][i]
        y0 = -1.257172*11.2+b
        line0 = (DF['ra'][i], DF['dec'][i], 11.2,y0)
        line1 = (0, 27.83644, 10.6847929,41.2690650)
        s1, s2 = fun2(line0, line1)
        l1 = ((s1-DF['ra'][i])**2+(s2-DF['dec'][i])**2) / (np.cos(math.radians(74))**2)
        l0 = np.sqrt(l1+(s1-10.6847929)**2+(s2-41.2690650)**2)
        length.append(np.sqrt(l0))
    DF['axis_length'] = length

    unbins = []
    dff2 = DF.groupby(by='brick')
    labels = [i for i in range(20)]
    for i in dff2:
        dd = i[1]
        dd = dd.sort_values('axis_length').reset_index(drop=True)
        cut = pd.qcut(dd.axis_length, 20, labels=labels)
        dd['wbin'] = cut
        dd = dd.sort_values(by='width', ascending=True).reset_index(drop=True)
        dd = dd.groupby(by=['wbin']).head(len(dd)//100)
        unbins.append(dd)
    grouped = pd.concat([i for i in unbins])
    grouped = grouped.sort_values('density2').reset_index(drop=True)

    zp = np.polyfit(grouped['density2'], grouped['cshift'], 1)
    zy = np.poly1d(zp)
    css = zy(grouped['density2'])
    std = np.std(css)

    grouped['redshift_compare'] = css+0.5*std
    grouped['blueshift_compare'] = css-7*std
    df_low = pd.DataFrame(yl)
    df_high = pd.DataFrame(yh)
    grouped['width_compare'] = pd.concat([df_low, df_high]).reset_index(drop=True)
    unred = grouped[(grouped['blueshift_compare'] < grouped['cshift']) & (grouped['cshift'] <grouped['redshift_compare']) & 
                    (grouped['width'] < grouped['width_compare'])]

    return unred


raw_rgb = cal_RGB()
rgb_with_density = cal_density(raw_rgb)
reducted_rgb = mag_cut(rgb_with_density)
unred_rgb = cal_unred_rgb(reducted_rgb)

c = open(unred_rgb_path, 'w')
pickle.dump(unred_rgb, c)
c.close()

c = open(rgb_path, 'w')  # unred_rgb_cmd
pickle.dump(reducted_rgb, c)  # data_cmd
c.close()

