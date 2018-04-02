# data_process

环境：Python3

raw_data: 存放测试数据，数据来源 https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data

## convertAvg.py
reference: https://github.com/kaz-Anova/StackNet/blob/master/example/twosigma_kaggle/create_files_v1.py

- 对高基数特征列统计每个特征值的频数(value_counts())，替换掉原先的特征列值，举例'a'列的其中一个特征值为'xx'，在该列中出现12次，12代替原先的'xx'；
- 初始化goods字典和bads字典，goods存放特征值对应的y值之和，bads存放特征值对应的y值的的频数。举例'a'列的其中一个特征值为12，对应的y值为[0,1,1]，则goods为(0+1+1)=2，bads为(1+1=1)=3，后验值为2/3=0.66，替换掉原先的值12；
- 使用交叉验证，假如将将训练集切分5份，4份用于训练得出的后验值代入验证集中，经过5轮后，5份验证集拼成一份完整的训练集；在使用全部的训练集训练得出的后验值，代入测试集，作为测试集特征列的值。

## colDistribution.py
reference: https://github.com/plantsgo/Rental-Listing-Inquiries/blob/master/script.py

- 初始化building_level字典，统计特征值所对应的y值分布。举例，某列的一个特征值为'XX'，对应的y值为[13, 3, 0]，表示y(low)=13，y(medium)=3，y(high)=0；
- 然后除以特征值对应的y值和（归一化项），即13+3+0=16，所以[13/16, 3/16, 0/16]。存为三个特征列col_low, col_medium, col_high；
- 使用交叉验证，假如将将训练集切分5份，4份用于训练得出的后验值代入验证集中，经过5轮后，所有特征值都有对应的后验值；在使用全部的训练集训练得出的后验值，代入测试集，作为测试集特征列的值；
- 注意，当特征值存在后验值时，以最后一次交叉验证得出的后验值为准。这里我对其进行改动，改为计算5次交叉验证得出的后验值的均值。

与convertAvg不同，convertAvg是统计该特征值的对应的y值所有类的分布值，而colDistribution则区分每类的分布值。举例low=0出现8次，medium=1出现6次，high=2出现1次，则convertAvg的goods/bads=(0*8+1*6+2*1)/(8+6+1)，值越小或越大，表示该值的纯度越高，即都属于同一类。而colDistribution则为[8, 6, 1]，表示y(low)出现8次，y(medium)出现6次，y(high)出现1次，然后除以总次数15，即[8/15, 6/15, 1/15]，使用三列来表示一列的特征值的分布。


# imbalanced_process
　　使用分类器对y值进行预测，通常情况下y > 0.5，判断为正例，否则为反例。y实际上表达了正例的可能性，几率 y/(1-y) 则反映了正例可能性与反例可能性的比值，阈值设置为0.5表明了分类器认为真实正、反例可能性相同，即分类器决策规则为 y/(1-y) > 1 则预测为正例。 </br >
　　然而，当训练集正、反例的数目不同时，令 m+ 表示正例数目，m- 表示反例数目，则观测几率为 m+ / m-，由于我们通常假设训练集是真实样本总体的无偏采样，因此观测几率就代表了真实几率。于是，只要分类器的预测几率高于观测几率就判定为正例，即 y/(1-y) > (m+ / m-)，则预测为正例。假设正负样本比例为1:10，则 m+ / m- = 0.1，于是只要正例与反例的几率比值 y/(1-y) 大于0.1，则判定为正例，实际上就是调整阈值。

## imbalanced_process.py
　　脚本调用imblearn包，展示了4种处理方法：
- 过采样，增加正例，使得正、反例数目相同，在使用学习器训练；缺点单纯复制正例，容易过拟合；
- 欠采样，减少反例，使得正、反例数目相同，在使用学习器训练。缺点丢失部分信息；
- SMOTE，通过对正例进行差值产生额外的正例，使得正、反例数目相同，在使用学习器训练；
- EasyEnsemble，假设将反例划分成10份，正例只有1份，训练10个学习器，每个学习器使用1份反例和1份正例，正例共用。在集合求结果，优点是不会丢失重要信息；
- 调整阈值，前面有提到；
- 代价敏感学习，部分算法内含参数class_weight，使用该参数调整正反例学习的代价，即调整正反例的权重；

reference：</br >
http://contrib.scikit-learn.org/imbalanced-learn/stable/user_guide.html </br >
《机器学习 - 周志华》
