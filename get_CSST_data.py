def get_csst_data():
    datalist = []
    columnlist = []
    with open('rgb_unred_csst_flag', 'r') as of:
        # 获取第一行表头数据
        firstline = of.readline()
        # 删除字符串头尾的特定字符
        firstline = firstline.strip('\n')
        # 将字符串按照空格进行分割
        columnlist = firstline.split()
        for i in of:
            i = i.strip('\n')
            # 将字符串按照空格进行分割
            i2 = i.split()
            datalist.append(i2)
    data = np.array(datalist).T
    s = []
    for i in data:
        s1 = []
        for j in i:
            j = float(j)
            s1.append(j)
        s.append(s1)
    csst = pd.DataFrame({
        columnlist[0]: s[0],
        columnlist[1]: s[1],
        columnlist[2]: s[2],
        columnlist[3]: s[3],
        columnlist[4]: s[4],
        columnlist[5]: s[5],
        columnlist[6]: s[6],
        columnlist[7]: s[7],
        columnlist[8]: s[8],
        columnlist[9]: s[9]
    })
    csst = csst[csst['Flag'] == 1.0]
    min1 = min(csst['RA'])
    max1 = max(csst['RA'])
    min2 = min(csst['DEC'])
    max2 = max(csst['DEC'])

    xbin = np.arange(min1, max1 + 0.004166666666666667, 0.004166666666666667)
    ybin = np.arange(min2, max2 + 0.004166666666666667, 0.004166666666666667)
    label1 = [k for k in range(len(xbin) - 1)]
    label2 = [k for k in range(len(ybin) - 1)]
    cut1 = pd.cut(csst['RA'], xbin, right=False, labels=label1)
    cut2 = pd.cut(csst['DEC'], ybin, right=False, labels=label2)

    xbin2 = np.arange(min1, max1 + 0.0019444444444444444, 0.0019444444444444444)
    ybin2 = np.arange(min2, max2 + 0.0019444444444444444, 0.0019444444444444444)
    labels1 = [k for k in range(len(xbin2) - 1)]
    labels2 = [k for k in range(len(ybin2) - 1)]
    cutt1 = pd.cut(csst['RA'], xbin2, right=False, labels=labels1)
    cutt2 = pd.cut(csst['DEC'], ybin2, right=False, labels=labels2)

    csst['xbin'] = cut1  #cal density
    csst['ybin'] = cut2
    csst['xbin2'] = cutt1  #mcmc
    csst['ybin2'] = cutt2

    csst['density'] = np.log10((csst.groupby(by=['xbin', 'ybin'])['xbin'].transform('count')) / 225)
    return csst

unred = get_csst_data().reset_index(drop=True) 

# def get_extinc(ebv):
#     wvl = np.array([4866,6215,7545,8679,9633])
#     flux = np.array([1,1,1,1,1])
#     fluxUnred = pyasl.unred(wvl, flux, ebv=ebv, R_V=3.1)
#     a=2.5*np.log(fluxUnred/flux)
#     return a
def get_extinc(av):
    co = [1.1426, 0.8519, 0.6455, 0.4903, 0.409]
    return [av * i for i in co]


# YUJIAO
def get_extinc(av):
    co = [1.1426, 0.8519, 0.6455, 0.4903, 0.409]
    return [av * i for i in co]


def add_av(time, num, bands):  #grizy

    magname = ['g_sn', 'r_sn', 'i_sn', 'z_sn', 'y_sn']
    #     co = [1.1426,0.8519,0.6455,0.4903,0.409]
    dd = open('./df_av/dfTime' + str(num) + '+' + str(time) + 's/df_sn' + str(num) +
        '+' + str(time) + 's_' + bands + '.pkl', 'rb') #csst_data_path
    #     dd=open('./df_av/dfTime' + str(num) + '+' + str(time) + 's/df_sn' + str(num) +
    #             '+'+ str(time) +'s.pkl','rb')
    csst = pickle.load(dd)
    dd.close()
    dff = csst.groupby(by=['xbin2', 'ybin2'])
    k = 0
    e = []

    for i in dff:
        e.append(i[1])

    av = [0.5, 0.5, 2.0, 2.0, 4.0, 4.0]
    sigma = [0.1, 0.3, 0.1, 0.3, 0.1, 0.3]  #6个组合
    for ii in range(6):
        exc = []
        for i in e:  #对每一块7*7做
            #             fred = random.uniform(0.1,0.9)
            df_red = i.sample(frac=0.6, replace=False, axis=0)  #抽出红化的
            d = pd.concat([i, df_red])
            df_unred = d.drop_duplicates(keep=False)  #未红化的
            avv = np.random.lognormal(np.log(av[ii]), sigma[ii], len(df_red))
            a_bands = np.array([get_extinc(i2) for i2 in avv]).T

            for j in range(5):
                aj = a_bands[j]
                df_red['A' + magname[j]] = aj
                df_red[magname[j] + '_av'] = df_red['A' + magname[j]] + df_red[magname[j]]  #每个band的消光后的星等mag_az
                df_unred['A' + magname[j]] = 0 * len(df_unred)
                df_unred[magname[j] + '_av'] = df_unred[magname[j]]
            df_red['Av'] = avv
            df_unred['Av'] = [0] * len(df_unred)
            df = pd.concat([df_red, df_unred]).reset_index(drop=True)
            exc.append(df)
        df2 = pd.concat([i for i in exc])
        df2['sigma'] = [sigma[ii]] * len(df2)
        print("写入：" + './model/time%s' % num + '+' + str(time) +
              's/mcmc%s' % ii + '/tdata%s' % ii + '/df_sn' + str(num) + '+' +
              str(time) + 's_' + bands + '_av' + str(ii) + '' + '.pkl')
        dd = open(
            './model/time%s' % num + '+' + str(time) + 's/mcmc%s' % ii +
            '/tdata%s' % ii + '/df_sn' + str(num) + '+' + str(time) + 's_' +
            bands + '_av' + str(ii) + '' + '.pkl', 'wb')  # df_csst_av_path
        #         
        pickle.dump(df2, dd)
        dd.close()
    return 0
ex_time = [150, 150, 150, 250, 250, 250, 250, 250]  # exposure time, typical time for main survey is 150s
ex_num = [1, 2, 4, 1, 2, 4, 8, 10]
for i in range(1, 2):
    for j in ['gz']:
        add_av(ex_time[i], ex_num[i], j)