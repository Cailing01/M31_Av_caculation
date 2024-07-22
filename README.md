# 消光计算
利用哈勃空间望远镜的近红外数据，进行近邻星系的消光计算。

# 方法原理
1、该方法由Dalcanton等人（2015）首次提出。
2、红巨星在近红外波段的CMD图（颜色星等图）上的分布应该是一条很窄的序列，受到消光影响的红巨星会向右移动（红化），而没有受到消光影响的则保持不动，这就导致这个序列变宽，甚至变成双峰。
3、我们希望通过衡量这个变化的程度，来计算消光值。具体做法就是找到该序列中未红化的红巨星，以及红化的红巨星，通过模型拟合两类红巨星的CMD图的差异，得到消光值。
4、我们选择的是贝叶斯模型。

# 注
我们利用该程序得到的是某一个pixel区域的消光值（这个pixel是由我们所决定的，如直径14角秒或7角秒），而不是某一颗恒星的消光值。

# 我们从拿到光学数据，到得到某个消光值的过程如下。
一、未红化和红化的红巨星列表
1、我们对Williams等人（2014）处理好的M31的星表，进行一些简单的筛选和计算，得到所有红巨星的列表以及红巨星的某些参数（如F110W、F160W波段的星等，所在pixel的宽度等）。通过某些参数的筛选，我们得到整个数据中未红化的红巨星的列表。（PHAT_rgb_reduction.py）
2、将整个红巨星列表按照空间位置进行分割（我们选择的是直径为7角秒的pixel）。对于每一个bin，我们通过密度最近原则为它分配相应的未红化的红巨星的列表。这一步的原因是我们无法直接区分出哪一颗红巨星是否被红化了，因此我们要人为地去为每一个bin分配未红化的红巨星序列。（PHAT_get_tdata.py）

二、模型拟合
对于每一个pixel的贝叶斯模型，我们设置四个参数，其中就包括消光值。然后我们通过emcee程序进行采样，得到最佳的参数值。（PHAT_model.py，PHAT_test.py）

三、分子云消光的计算
分子云的消光值一般要高于周围区域，因此该程序被寄希望于寻找近邻星系中的分子云。
这一点需要慎重考虑，因为我们的消光值是基于尘埃的影响，而尘埃的存在是否意味着分子云的存在还有待论证。

# CSST应用

一、简介
CSST具有超大的视场，与哈勃望远镜相当的分辨率，因此得到近邻星系的大量数据，进行类似的研究。
该方法很受波段的影响，我们在哈勃的数据中用的是1100nm和1600nm的数据，CSST的最长波段为y波段（963nm），因此我们要测试该方法在CSST波段的可行性。

二、具体实现
1、我们请国家天文台的汤静老师为我们实现了哈勃数据到CSST数据的转换，即将哈勃数据中筛选出的未红化的红巨星转化为CSST波段的数据。这些红巨星都没有受到消光的影响。
2、对这些未红化的红巨星，我们对其施加相应的噪声（模拟现实中的噪声）。
3、我们将这些红巨星均匀地撒在各个区域，对每个区域施加消光值Av0，并进行极限星等的筛选（某些红巨星会受到消光的影响而不可见）。
4、我们利用该方法去计算每个区域的消光值Av，通过对比Av以及Av0的差，去衡量该方法在CSST波段的数据中的可信性和准确度。
以上步骤在CSST_get_tdata.py,CSST_limit_cut.py实现。

# 结果
1、利用哈勃数据得到的结果，我们成功复现了M31的光学消光图，它与之前的研究所得的结果一致。
2、在模拟数据中，CSST的测试效果良好。

# 参考文献
Chen et al.(2024) doi:10.48550/arXiv.2405.19710 \\n
Dalcanton et al.(2015) doi:10.1088/0004-637X/814/1/3

