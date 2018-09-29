import csv
import jieba
import jieba.posseg as pseg
import re, os

import code

root_path = os.getcwd() + os.sep
c_mark = ['；', '。', '?', '？', '!', '！', ';']
slots = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW', 'CL']

cooked_corpus_path = root_path + "assets" + os.sep + "cooked_corpus"
raw_corpus_path = root_path + "assets" + os.sep + "raw_corpus"
dict_path = os.path.join(raw_corpus_path, 'DICT_NOW.csv')
original_files_path = raw_corpus_path + os.sep + "original_data" + os.sep

def gen_cooked_corpus():
    fout1 = open(os.path.join(cooked_corpus_path, 'example.dev'), 'w', encoding='utf8')
    fout2 = open(os.path.join(cooked_corpus_path, 'example.test'), 'w', encoding='utf8')
    fout3 = open(os.path.join(cooked_corpus_path, 'example.train'), 'w', encoding='utf8')

    flag = 0
    dics = csv.reader(open(dict_path, 'r', encoding='utf8'))
    for row in dics: # 增加字典
        if flag == 0:
            flag = 1
            continue
        if len(row) == 2:
            jieba.add_word(row[0].strip(), tag = row[1].strip())
    
    intents = os.listdir(original_files_path)

    for intentname in intents:
        intentpath = original_files_path + intentname + os.sep
        files = os.listdir(intentpath)

        split_num = 0
        for file in files:
            # 只处理文件名包含 txtoriginal 的文件
            if "txtoriginal" in file:
                fp = open(intentpath + file, 'r', encoding='utf8')
                lines=[line for line in fp]
                for line in lines:
                    split_num += 1
                    # jieba分词
                    words = pseg.cut(line)
                    for key, value in words:
                        if value.strip() and key.strip():
                            if value not in slots:
                                value='O'
                                for achar in key.strip():
                                    # 按行数来划分数据集，比例为 1:2:13
                                    if split_num % 15 < 2:
                                        index = str(1)
                                    elif split_num % 15 > 1 and split_num % 15 < 4:
                                        index = str(2)
                                    else:
                                        index = str(3)

                                    if achar and achar.strip() in c_mark: # 如果是标点符号就多换一个行
                                        string = achar + " " + value.strip() + " " + intentname + "\n" + "\n"
                                    
                                        if index == '1':                               
                                            fout1.write(string)
                                        elif index == '2':
                                            fout2.write(string)
                                        elif index == '3':
                                            fout3.write(string)
                                        else:
                                            pass
                                    elif achar.strip() and achar.strip() not in c_mark:
                                        string = achar + " " + value.strip() + " " + intentname + "\n"

                                        if index == '1':
                                            fout1.write(string)
                                        elif index == '2':
                                            fout2.write(string)
                                        elif index == '3':
                                            fout3.write(string)
                                        else:
                                            pass
                                    else:
                                        continue
                            elif value.strip() in slots:
                                begin = 0
                                for char in key.strip():
                                    if begin == 0:
                                        begin += 1
                                        string1 = char + ' ' + 'B-' + value.strip() + " " + intentname + '\n'

                                        if index == '1':
                                            fout1.write(string1)
                                        elif index == '2':
                                            fout2.write(string1)
                                        elif index == '3':
                                            fout3.write(string1)
                                        else:
                                            pass
                                    else:
                                        string1 = char + ' ' + 'I-' + value.strip() + " " + intentname + '\n'

                                        if index == '1':
                                            fout1.write(string1)
                                        elif index == '2':
                                            fout2.write(string1)
                                        elif index == '3':
                                            fout3.write(string1)
                                        else:
                                            pass
                            else:
                                continue
                            
    fout1.close()
    fout2.close()
    fout3.close()
