import csv
import jieba
import jieba.posseg as pseg
import re
import os

import json
import codecs

root_path = os.getcwd() + os.sep

slots = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT',
         'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW', 'CL']

raw_corpus_path = root_path + "assets" + os.sep + "raw_corpus"
dict_path = os.path.join(raw_corpus_path, 'DICT_NOW.csv')
original_files_path = raw_corpus_path + os.sep + "original_data" + os.sep



def findEntity(text, search, entity, start=0):
    result = []
    while True:
        index = text.find(search, start)
        if index == -1:
            break

        start = index + 1

        result.append({
            "start": start - 1,
            "end": start - 1 + len(search),
            "value": search,
            "entity": entity
        })

    return result


def gen_cooked_corpus():
    flag = 0
    dics = csv.reader(open(dict_path, 'r', encoding='utf8'))

    for row in dics:  # 增加字典
        if flag == 0:
            flag = 1
            continue
        if len(row) == 2:
            jieba.add_word(row[0].strip(), tag=row[1].strip())

    intents = os.listdir(original_files_path)

    rasa_data = {
        "regex_features": [],
        "entity_synonyms": [],
        "common_examples": []
    }

    common_examples = []

    for intentname in intents:
        intentpath = original_files_path + intentname + os.sep
        files = os.listdir(intentpath)

        for file in files:
            # 只处理文件名包含 txtoriginal 的文件
            if "txtoriginal" in file:
                fp = open(intentpath + file, 'r', encoding='utf8')
                lines = [line for line in fp]
                
                entity_example = {
                    "text": "",
                    "intent": "",
                    "entities": []
                }

                for line in lines:
                    # jieba分词
                    if line.strip():
                        entity_example["text"] = line.strip()
                        entity_example["intent"] = intentname

                        words = pseg.cut(line)
                        for key, value in words:
                            if value.strip() and key.strip():
                                if value.strip() in slots:
                                    rt = findEntity(line.strip(), key.strip(), value.strip())
                                    entity_example["entities"].extend(rt)
                                else:
                                    continue

                common_examples.append(entity_example)

    rasa_data["common_examples"] = common_examples


    return rasa_data


rasa_data = gen_cooked_corpus()


json_text = json.dumps(rasa_data)

with codecs.open('1.json', 'w', 'utf-8') as f:
    f.write(json_text)
