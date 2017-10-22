# data_process

环境：Python3

raw_data: 存放测试数据，数据来源 https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data

## convertAvg.py
参考 https://github.com/kaz-Anova/StackNet/blob/master/example/twosigma_kaggle/create_files_v1.py

- 对高基数特征列统计每个特征值的频数(value_counts())，替换掉原先的特征列值，举例'a'列的其中一个特征值为'xx'，在该列中出现12次，12代替原先的'xx'；
- 初始化goods字典和bads字典，goods存放特征值对应的y值之和，bads存放特征值对应的y值的的频数。举例'a'列的其中一个特征值为12，对应的y值为[0,1,1]，则goods为(0+1+1)=2，bads为(1+1=1)=3，后验值为2/3=0.66，替换掉原先的值12；
- 使用交叉验证，假如将将训练集切分5份，4份用于训练得出的后验值代入验证集中，经过5轮后，5份验证集拼成一份完整的训练集；在使用全部的训练集训练得出的后验值，代入测试集，作为测试集特征列的值。

## colDistribution.py
参考 https://github.com/plantsgo/Rental-Listing-Inquiries/blob/master/script.py

- 初始化building_level字典，统计特征值所对应的y值分布。举例，某列的一个特征值为'XX'，对应的y值为[13, 3, 0]，表示y(low)=13，y(medium)=3，y(high)=0；
- 然后除以特征值对应的y值和（归一化项），即13+3+0=16，所以[13/16, 3/16, 0/16]。存为三个特征列col_low, col_medium, col_high；
- 使用交叉验证，假如将将训练集切分5份，4份用于训练得出的后验值代入验证集中，经过5轮后，所有特征值都有对应的后验值；在使用全部的训练集训练得出的后验值，代入测试集，作为测试集特征列的值；
- 注意，当特征值存在后验值时，以最后一次交叉验证得出的后验值为准。这里我对其进行改动，改为计算5次交叉验证得出的后验值的均值。

与convertAvg不同，convertAvg是统计该特征值的对应的y值所有类的分布值，而colDistribution则区分每类的分布值。举例low=0出现8次，medium=1出现6次，high=2出现1次，则convertAvg的goods/bads=(0*8+1*6+2*1)/(8+6+1)，值越小或越大，表示该值的纯度越高，即都属于同一类。而colDistribution则为[8, 6, 1]，表示y(low)出现8次，y(medium)出现6次，y(high)出现1次，然后除以总次数15，即[8/15, 6/15, 1/15]，使用三列来表示一列的特征值的分布。


## text_regex.py
参考 https://github.com/ChenglongChen/Kaggle_HomeDepot/blob/master/Code/Chenglong/data_processor.py

使用基类+正则化函数构建一系列正则化功能，对文本进行一步步的预处理：
- 移除多个空格符和5个字符及以上的大写字符或数字；
- 将所有字符转为小写，如'XX' -> 'xx'；
- 统一格式，'4x8wood paneling' -> '4 x 8 wood paneling'；
- 转为统一单位，如'6ft h bamboo fencing' -> '6 ft. h bamboo fencing'；
- 统一数字格式，1,000,000 -> 1000000，one -> 1，two -> 2。
