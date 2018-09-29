import json
import codecs
import os

root_path = os.getcwd() + os.sep
c_mark = ['；', '。', '?', '？', '!', '！', ';']
cooked_corpus_path = root_path + "assets" + os.sep + "cooked_corpus"


with codecs.open('1.json', 'r', 'utf-8') as f:
    json_dict = json.load(f)

fout1 = open(os.path.join(cooked_corpus_path,'demo.dev'), 'w', encoding='utf8')


for value in json_dict.get("common_examples", []):
    text = value.get("text")
    entities = value.get("entities")
    value = 'O'

    bilou = [value for _ in text]

    for item in entities:
        start = item.get("start")
        end = item.get("end")
        entity = item.get("entity")

        if start is not None and end is not None:
            bilou[start] = 'B-' + entity
            for i in range(start+1, end):
                bilou[i] = 'I-' + entity

    for index, achar in enumerate(text):
        if achar and achar.strip() in c_mark: # 如果是标点符号就多换一个行
            string = achar + " " + bilou[index] + "\n" + "\n"
            fout1.write(string)

        elif achar.strip() and achar.strip() not in c_mark:
            string = achar + " " + bilou[index] + "\n"
            fout1.write(string)

        else:
            continue



fout1.close()
