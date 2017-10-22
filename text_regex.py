#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
https://github.com/ChenglongChen/Kaggle_HomeDepot/blob/master/Code/Chenglong/data_processor.py
'''

import regex
import pandas as pd

#--------------------------- Processor ---------------------------
# 基本类，通过对pattern_replace_pair_list赋予不同的正则函数，进行正则处理
# 大部分处理可以被转换成'pattern-replace'框架
class BaseReplacer:
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list
    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return regex.sub(r'\s+', ' ', text).strip()


###############################################################################
# 将大写转成小写
class LowerCaseConverter(BaseReplacer):
    '''
    如Traditional -> traditional
    '''
    def transform(self, text):
        return text.lower()


###############################################################################
# 处理数字
class DigitLetterSplitter(BaseReplacer):
    '''
    x:
    1x1x1x1x1 -> 1 x 1 x 1 x 1 x 1
    19.875x31.5x1 -> 19.875 x 31.5 x 1

    -:
    1-Gang -> 1 Gang
    48-Light -> 48 Light

    .:
    simplify installation.60 in. L x 36 in. W x 20 in. ->
    simplify installation. 60 in. L x 36 in. W x 20 in.
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r'(\d+)[\.\-]*([a-zA-Z]+)', r'\1 \2'),
            (r'([a-zA-Z]+)[\.\-]*(\d+)', r'\1 \2'),
        ]


###############################################################################
class DigitCommaDigitMerger(BaseReplacer):
    '''
    1,000,000 -> 1000000
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r'(?<=\d+),(?=000)', r''),
        ]


###############################################################################
class NumberDigitMapper(BaseReplacer):
    '''
    one -> 1
    two -> 2
    '''
    def __init__(self):
        numbers = [
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
            'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90
        ]
        self.pattern_replace_pair_list = [
            # regex.sub(r'(?<=\W|^)zero(?=\W|$)', '0', 'xx zero aa')
            (r'(?<=\W|^){:s}(?=\W|$)'.format(n), str(d)) for n, d in zip(numbers, digits)
        ]


###############################################################################
# 转换成统一单位
class UnitConverter(BaseReplacer):
    '''
    shadeMature height: 36 in. - 48 in.Mature width
    PUT one UnitConverter before LowerUpperCaseSplitter
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
            # inches|inch|in|in.|都转换成in. 如'xx 546 inches'转换为'xx 546 in. '
            (r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r'\1 in. '),
            (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r'\1 lb. '),
            (r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r'\1 ft. '),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r'\1 sq.in. '),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r'\1 sq.ft. '),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r'\1 cu.in. '),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r'\1 cu.ft. '),
            (r"([0-9]+)( *)(gallons|gallon|gal)\.?", r'\1 gal. '),
            (r"([0-9]+)( *)(ounces|ounce|oz)\.?", r'\1 oz. '),
            (r"([0-9]+)( *)(centimeters|cm)\.?", r'\1 cm. '),
            (r"([0-9]+)( *)(milimeters|mm)\.?", r'\1 mm. '),
            (r"([0-9]+)( *)(minutes|minute)\.?", r'\1 min. '),
            (r"([0-9]+)( *)(°|degrees|degree)\.?", r'\1 deg. '),
            (r"([0-9]+)( *)(v|volts|volt)\.?", r'\1 volt. '),
            (r"([0-9]+)( *)(wattage|watts|watt)\.?", r'\1 watt. '),
            (r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r'\1 amp. '),
            (r"([0-9]+)( *)(qquart|quart)\.?", r'\1 qt. '),
            (r"([0-9]+)( *)(hours|hour|hrs.)\.?", r'\1 hr '),
            (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r'\1 gal. per min. '),
            (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r'\1 gal. per hr '),
        ]


#------------------- Extract Product Name -------------------
# 对于CrowdFlower的解决方案
color_list = ['white', 'black', 'brown', 'gray', 'chrome', 'stainless steel', 'whites',
              'red', 'browns / tans', 'bronze', 'silver', 'blacks', 'beige', 'stainless',
              'blue', 'nickel', 'metallics', 'clear', 'grays', 'green', 'multi', 'beige / cream',
              'tan', 'greens', 'yellow', 'wood', 'blues', 'reds / pinks', 'brushed nickel',
              'orange', 'metallic', 'brass', 'yellows / golds', 'oil rubbed bronze',
              'polished chrome', 'almond', 'multi-colored', 'dark brown wood', 'primed white',
              'beige/bisque', 'biscuit', 'ivory', 'oranges / peaches', 'grey', 'unfinished wood',
              'light brown wood', 'wood grain', 'silver metallic', 'copper', 'medium brown wood',
              'soft white', 'gold', 'satin nickel', 'cherry', 'bright white', 'red/orange', 'teal',
              'natural', 'oak', 'mahogany', 'aluminum', 'espresso', 'unfinished', 'purples / lavenders',
              'brown/tan', 'steel', 'venetian bronze', 'slate', 'warm white', 'bone', 'pink', 'stainless look',
              'reddish brown wood', 'solid colors', 'off-white', 'walnut', 'chocolate', 'light almond',
              'vibrant brushed nickel', 'satin white', 'polished brass', 'linen','white primer', 'purple',
              'charcoal', 'color', 'oil-rubbed bronze', 'melamine white', 'turquoises / aquas', 'blue/purple',
              'primed', 'bisque', 'browns/tans', 'assorted colors', 'java', 'pewter', 'chestnut', 'yellow/gold',
              'taupe', 'pacific white', 'cedar', 'monochromatic stainless steel', 'other', 'platinum',
              'mocha', 'cream', 'sand', 'daylight', 'brushed stainless steel', 'powder-coat white',]
# color_list为各种颜色的列表
colors_pattern = r'(?<=\W|^){:s}(?=\W|$)'.format('|'.join(color_list))
# ['in.', 'lb.', 'ft.', 'sq.in.', 'sq.ft.'...]
units = [' '.join(r.strip().split(' ')[1:]) for p, r in UnitConverter().pattern_replace_pair_list]


###############################################################################
# 构建一系列正则化处理功能
class ProductNameExtractor(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            # Remove descriptions (text between paranthesis/brackets)
            ('[ ]?[[(].+?[])]', r''),
            # 移除'made in...'
            ('made in [a-z]+\\b', r''),
            # Remove descriptions (hyphen or comma followed by space then at most 2 words, repeated)
            ('([,-]( ([a-zA-Z0-9]+\\b)){1,2}[ ]?){1,}$', r''),
            # 移除描述中介词及介词后面的内容，介词with, for, by, in，即with xx被移除
            ('\\b(with|for|by|in|w/) .+$', r''),
            # 移除'size: ...'
            ('size: .+$', r''),
            # 移除'size 数字'，包括小数，如size 44.99被移除
            ('size [0-9]+[.]?[0-9]+\\b', r''),
            # 移除各种颜色，如'aa white bb black'移除后为'aa bb'
            (colors_pattern, r''),
            # 移除以下词汇
            ('(value bundle|warranty|brand new|excellent condition|one size|new in box|authentic|as is)', r''),
            # 移除停用词'in'
            ('\\b(in)\\b', r''),
            # 格式为'aa-aa'的相同词进行合并，为'aaaa'
            ('([a-zA-Z])-([a-zA-Z])', r'\1\2'),
            # 移除特殊字符，如&<>)(_,.;:!?/+#*-
            ('[ &<>)(_,.;:!?/+#*-]+', r' '),
            # 移除只有数字的，如'98'处理后为''
            ('\\b[0-9]+\\b', r''),
        ]

    # 文本处理
    def preprocess(self, text):
        pattern_replace_pair_list = [
            # 移除多个空格符，变成一个空格符，如'aa  b' -> 'aa b'
            ('[\"]+', r''),
            # 移除5个字符及以上的大写字符或数字，如'ABCDE  12345 bb'处理后为' bb'
            ('[ ]?\\b[0-9A-Z-]{5,}\\b', ''),
        ]
        # 移除多个空格符和5个字符及以上的大写字符或数字
        text = BaseReplacer(pattern_replace_pair_list).transform(text)
        # 将所有字符转为小写，如'XX' -> 'xx'
        text = LowerCaseConverter().transform(text)
        # '4x8wood paneling' -> '4 x 8 wood paneling'
        text = DigitLetterSplitter().transform(text)
        # 转为统一单位，如'6ft h bamboo fencing' -> '6 ft. h bamboo fencing'
        text = UnitConverter().transform(text)
        # 1,000,000 -> 1000000
        text = DigitCommaDigitMerger().transform(text)
        # one -> 1，two -> 2
        text = NumberDigitMapper().transform(text)

        return text

    # 对每个文本进行一系列处理
    def transform(self, text):
        text = super().transform(self.preprocess(text))

        return text


###############################################################################
if __name__ == '__mian__':
    file_path = '../data/'
    data = pd.read_csv(file_path + 'train.csv', encoding='ISO-8859-1')

    ext = ProductNameExtractor()
    data['search_term_product_name'] = data['search_term'].apply(ext.transform)

    ################################################################################
    # 通过替换pattern_replace_pair_list，查看单个正则函数的效果
    pattern_replace_pair_list = [(r'(\d+)[\.\-]*([a-zA-Z]+)', r'\1 \2'),
                                 (r'([a-zA-Z]+)[\.\-]*(\d+)', r'\1 \2'),]
    func = BaseReplacer(pattern_replace_pair_list)
    data['re_text'] = data['search_term'].map(lambda x: func.transform(x))
    flag = data['re_text'] != data['search_term']
    #flag.value_counts()

    data.loc[flag==True, ['re_text', 'search_term']].head(10)
