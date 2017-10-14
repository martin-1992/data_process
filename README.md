# data_process

环境：Python3

raw_data: 存放测试数据，数据来源 https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data

## convertAvg.py
参考 https://github.com/kaz-Anova/StackNet/blob/master/example/twosigma_kaggle/create_files_v1.py

- 对高基数特征列统计每个特征值的频数(value_counts())，替换掉原先的特征列值；
- 

计算goods/bads来计算后验值，其中goods为某列的特征值对应的y值之和，bads为某列的特征值对应的y值的频数，举例'a'列的其中一个特征值为'xx'，对应的y值为[0,1,1]，则goods为(0+1+1)=2，bads为(1+1=1)=3，后验值为2/3=0.66。


