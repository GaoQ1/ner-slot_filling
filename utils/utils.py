import os
import json
import shutil
import logging

import tensorflow as tf
import numpy as np
from utils.conlleval import return_report

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w", encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    
    eval_lines = return_report(output_file)

    # code.interact(local=locals())
    return eval_lines

def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))

def make_path(params): #创建 path
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")

def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)
        os.mkdir(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)
        os.mkdir(params.summary_path)

    if os.path.isdir(params.configs_path):
        shutil.rmtree(params.configs_path)
        os.mkdir(params.configs_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)
        os.mkdir(params.result_path)

    # if os.path.isdir("assets/cooked_corpus"):
    #     shutil.rmtree("assets/cooked_corpus")
    #     os.mkdir("assets/cooked_corpus")

    if os.path.isdir("log"):
        shutil.rmtree("log")
        os.mkdir("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)

def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)

def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")

def create_model(session, Model_class, path, load_vec, config, id_to_char, logger):
    # create model, reuse parameters if exists
    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        # 加载模型默认参数与预处理的词嵌入
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            # 加载模型初始化生成的char_lookup参数值
            emb_weights = session.run(model.char_lookup.read_value())
            # 更新词嵌入权重：如果char在预训练的词嵌入集中存在便替换
            emb_weights = load_vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)

            session.run(model.char_lookup.assign(emb_weights)) # 重新赋值
            logger.info("Load pre-trained embedding.")
    return model


def result_to_json(string, tags, intentName, probability):
    item = {
        "string": string,
        "entities": [],
        "intent": {}
    }
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    item["intent"] = {
        "intentName": intentName,
        "probability": probability
    }    
    return item


def accuracy_score(true_data, pred_data, true_length = None):
    true_data = np.array(true_data)
    pred_data = np.array(pred_data)
    assert true_data.shape == pred_data.shape

    if true_length is not None:
        val_num = np.sum(true_length)
        assert val_num != 0
        res = 0
        for i in range(true_data.shape[0]):
            res += np.sum(true_data[i, :true_length[i]]
                          == pred_data[i, :true_length[i]])
    else:
        val_num = np.prod(true_data.shape) # 返回给定轴上数据的乘积
        assert val_num != 0
        res = np.sum(true_data == pred_data)
    res /= float(val_num)
    return res
