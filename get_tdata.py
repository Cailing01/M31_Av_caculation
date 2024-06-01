import pandas as pd
import numpy as np
import multiprocessing
import time
import os
import sys
import emcee
import corner
import pickle
import math

ra0 = 10.6847929 + (11.6 - 10.6847929) * np.sin(math.radians(37))
dec0 = (11.6 - 10.6847929) * np.cos(math.radians(37)) + 41.2690650
slope = (dec0 - 41.2690650) / (ra0 - 10.6847929)
b = dec0 - slope * ra0

c = open('Draine_et_al2014_Av.pkl', 'rb')
draine = pickle.load(c)
c.close()


def distances(point0, point1):
    A = dec0 - 41.2690650
    B = 10.6847929 - ra0
    C = (41.2690650 - dec0) * 10.6847929 + (ra0 - 10.6847929) * 41.2690650
    distance = np.abs(A * point0 + B * point1 + C) / (np.sqrt(A**2 + B**2))
    return distance


def get_tdata(df_rgb_path):
    # df_RGB
    cc = open(df_rgb_path, 'rb')
    df = pickle.load(cc).reset_index(drop=True)
    cc.close()

    # df_unred_RGB
    cc = open('unred_rgb_path', 'rb')
    unred = pickle.load(cc).reset_index(drop=True)
    cc.close()

    # cal mfred
    leng = []
    for i in range(len(df)):
        j = distances(df['ra'][i], df['dec'][i])
        yy = df['ra'][i] * slope + b
        if yy > df['dec'][i]:
            mf = 0.5 - j**2
        else:
            mf = 0.5 + j**2
        leng.append(mf)
    df['mfred'] = leng

    # unred bins ( more than 2500 RGB)
    unred = unred.sort_values('density2')
    classi = unred['density2'].unique()

    dfclass = []
    result = []
    for i in classi:
        j = unred[unred['density2'].isin([i])]
        dfclass.append(j)

    packed = []
    l = -1
    for i in range(len(dfclass)):
        if i < l or i == l:
            i += 1
        else:
            k = np.mean(dfclass[i]['density2'])
            leng = len(dfclass[i])
            s = []
            s.append(dfclass[i])
            if (i == len(dfclass) - 1) or (leng >= 2500):
                packed.append(s)
            else:
                for j in range(i + 1, len(dfclass)):
                    delta = np.mean(dfclass[j]['density2']) - k
                    leng += len(dfclass[j])
                    s.append(dfclass[j])
                    if j == len(dfclass) - 1:
                        ss = pd.concat([i for i in s])
                        packed.append(ss)
                        l = j
                        break
                    if (delta >= 0.05) and (leng >= 2500):
                        ss = pd.concat([i for i in s])
                        packed.append(ss)
                        l = j
                        break
    packed2 = []
    packed3 = []
    for i in packed:
        if type(i) != pd.core.frame.DataFrame:
            i = i[0].copy()
        i['mdensity'] = [np.median(i['density2'])] * len(i)
        i['color'] = i['f110w'] - i['f160w']
        i['c0'] = 0.752 - 0.1 * (i['f160w'] - 22.0) - 0.006 * ((i['f160w'] - 22.0)**2)
        i['q'] = i['f160w'] - (i['f110w'] - i['f160w'] - i['c0']) * (0.2029 / (0.3266 - 0.2029))
        i = i.drop(['xbin', 'xbin2', 'ybin2', 'ybin'], axis=1)
        packed2.append(i)

    for i in packed2:  #unred- color 
        if type(i) != pd.core.frame.DataFrame:
            i = i[0].copy()
        o = np.mean(i['color'])
        k = 3.5 * (np.std(i['color'])) + o
        j = i[(i['color'] >= 0) & (i['color'] < k)]
        packed3.append(j)

    # data bins 7*7

    # df = df[df['dec'] > min(df['dec'])+0.0009722222222222222]
    min1 = min(df['ra'])
    max1 = max(df['ra'])
    min2 = min(df['dec'])
    max2 = max(df['dec'])

    xbin = np.arange(min1, max1 + 0.0019444444444444444, 0.0019444444444444444)
    ybin = np.arange(min2, max2 + 0.0019444444444444444, 0.0019444444444444444)
    labels1 = [k for k in range(len(xbin) - 1)]
    labels2 = [k for k in range(len(ybin) - 1)]
    cut1 = pd.cut(df['ra'], xbin, right=False, labels=labels1)
    cut2 = pd.cut(df['dec'], ybin, right=False, labels=labels2)

    df['xbin'] = cut1  #cal density
    df['ybin'] = cut2

    grouped = df.groupby(by=['xbin', 'ybin'])
    listed = []
    for i in grouped:
        listed.append(i[1])

    listed2 = []
    for i in listed:
        if type(i) != pd.core.frame.DataFrame:
            i = i[0].copy()
        i['mdensity'] = np.median(i['density2'])
        i['color'] = i['f110w'] - i['f160w']
        i['c0'] = 0.752 - 0.1 * (i['f160w'] - 22.0) - 0.006 * ((i['f160w'] - 22.0)**2)
        i['q'] = i['f160w'] - (i['f110w'] - i['f160w'] - i['c0']) * (0.2029 / (0.3266 - 0.2029))
        i['count'] = len(i)
        #         i=i.drop(['label'],axis=1)
        listed2.append(i)

#     data-unred match
    data = []
    for i in listed2:  #data
        if type(i) != pd.core.frame.DataFrame:
            i = i[0].copy()
        bb = []
        aa1 = []
        aa2 = []
        aa3 = []
        bb1 = []
        bb2 = []
        cc1 = []
        cc2 = []
        cc4 = []
        cc6 = []
        cc7 = []
        cc8 = []
        refer = 100
        k = packed3[0]
        for j in packed3:  #unred
            if type(j) != pd.core.frame.DataFrame:
                j = j[0]
            delden = np.abs(
                np.median(i['mdensity']) - np.median(j['mdensity']))
            if delden <= refer:
                refer = delden
                k = j
        for m in k['color']:
            m = np.float64(m)
            aa1.append(m)
        for n in k['q']:
            n = np.float64(n)
            aa2.append(n)
        for q in i['mfred']:
            aa3.append(q)
        for o in i['color']:
            o = np.float64(o)
            bb1.append(o)
        for p in i['q']:
            p = np.float64(p)
            bb2.append(p)
        for p in i['ra']:
            p = np.float64(p)
            cc1.append(p)
        for p in i['dec']:
            p = np.float64(p)
            cc2.append(p)
        for p in i['f160w']:
            p = np.float64(p)
            cc4.append(p)
        for p in i['f110w']:
            p = np.float64(p)
            cc6.append(p)
        for p in i['f110w_err']:
            p = np.float64(p)
            cc7.append(p)
        for p in i['f160w_err']:
            p = np.float64(p)
            cc8.append(p)

        bb.append([aa1, aa2])  # unred: color、q data[0]
        bb.append(aa3)  # unred: mfred data[1]
        bb.append([bb1, bb2])  # data: color、q data[2]
        bb.append([cc1, cc2, cc4, cc6, cc7, cc8])  # data的ra,dec,ymag,gmag data[3]
        data.append(bb)

    tdata = []

    for i in range(len(data)):
        d = []
        kk = [[], []]
        kk2 = [[], []]
        rdata = [[] for i in range(8)]
        l = 0
        for j in range(len(data[i][2][0])):  #data: color
            o = np.mean(data[i][2][0])
            k = 3.5 * (np.std(data[i][2][0])) + o
            k2 = 0.25 * (np.std(data[i][2][0]))
            if data[i][2][0][j] > k2 and data[i][2][1][j] > 18.5:
                kk2[0].append(data[i][2][0][j])
                kk2[1].append(data[i][2][1][j])
                if data[i][2][0][j] > k:
                    l += 1
                    kk[0].append(data[i][2][0][j])
                    kk[1].append(data[i][2][1][j])
                else:
                    rdata[0].append(data[i][2][0][j])
                    rdata[1].append(data[i][2][1][j])
                    rdata[2].append(data[i][3][0][j])
                    rdata[3].append(data[i][3][1][j])
                    rdata[4].append(data[i][3][2][j])
                    rdata[5].append(data[i][3][3][j])
                    rdata[6].append(data[i][3][4][j])
                    rdata[7].append(data[i][3][5][j])

        if len(kk2[0]) == 0:
            continue
        xmax = max(max(data[i][0][0]), max(data[i][2][0]))
        xmin = min(min(data[i][0][0]), min(data[i][2][0]))
        ymax = max(max(data[i][0][1]), max(data[i][2][1]))
        ymin = min(min(data[i][0][1]), min(data[i][2][1]))
        edge = []
        xedges = np.arange(xmin - 0.015, xmax + 0.015, 0.015)  #yes
        yedges = np.arange(ymin - 0.2, ymax + 0.2, 0.2)
        #         d.append([xedges,yedges])#v[7]
        #         edge.append([xedges,yedges])

        #     punred
        Hun, xedges1, yedges1 = np.histogram2d(data[i][0][0], data[i][0][1], bins = [xedges, yedges])
        Hun0 = (Hun.flatten() / sum(Hun.flatten()))
        shape = np.shape(Hun)

        #     pnoise
        fnoise = l / len(kk2[0])
        if l == 0:
            pnoise = np.array([0] * len(Hun0))
        else:
            pnoise1, xedges1, yedges1 = np.histogram2d(kk[0], kk[1], bins = [xedges, yedges])
            pnoise = pnoise1.flatten() / sum(pnoise1.flatten())

    # pdata
        Hdata, xedges2, yedges2 = np.histogram2d(rdata[0], rdata[1], bins = [xedges, yedges])
        Hdata0 = (Hdata.flatten() / sum(Hdata.flatten()))  #yes
        pdata = []
        for k in Hdata0:
            # if k == 0:
            #     k=0.00000000001
            pdata.append(k)
        pdata = np.array(pdata)
        df_data = pd.DataFrame({
            'ra': rdata[2],
            'dec': rdata[3],
            'f110w': rdata[5],
            'f160w': rdata[4],
            'color': rdata[0],
            'q': rdata[1],
            'f110w_err': rdata[6],
            'f160w_err': rdata[7]
        })

        d.append(i)
        d.append(np.median(data[i][1]))  # v[1] mfred
        d.append(shape)  #v[2]
        d.append(data[i][0])  #v[3] unred cmd
        d.append(df_data)  #v[4] data cmd、ra\dec、f160w
        d.append(Hun0)  #v[5] unred pdf
        d.append(fnoise)  # v[6]
        d.append(pnoise)  #v[7]
        d.append(pdata)  #v[6] data pdf
        d.append([xedges, yedges])  #v[7]
        #         tdata.append(len(kk2[0]))
        tdata.append(d)
    for i in range(len(tdata)):
        j = np.median(tdata[i][3][0])
        k = max(tdata[i][4]['color'])
        m = (k - j) / 0.124
        ra = np.mean(tdata[i][4]['ra'])
        dec = np.mean(tdata[i][4]['dec'])
        tdata[i].append(m)


#         draine2 = draine[(ra - 10/ 3600 <= draine['ra']) & (draine['ra'] <= ra + 10 / 3600) &
#                     (dec - 10 / 3600 <= draine['dec']) & (draine['dec'] <= dec + 10 / 3600)]
#         if len(draine2) == 0:
#             print(0/0)
#         avd = np.mean(draine2['Avdraine'])

#     print(len(tdata))

    return tdata

excel = []
filelist = [i for i in range(2, 24)]

for i in filelist:
    j = 'dff%i.pkl' % i
    j2 = tdata_path + 'tdata%i.pkl' % i
    excel.append([j, j2])

brick_num = 0
for i in excel:
    d = get_tdata(i[0], i[1])
    c = open(tdata_path + '/tdata%s' % filelist[brick_num] + '/tdata.pkl', 'wb')
    pickle.dump(d, c)
    c.close()
    brick_num += 1