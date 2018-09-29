import codecs
import pickle
import itertools
import os
from collections import OrderedDict
from gevent import monkey  # 多线程的库
monkey.patch_all()

import tensorflow as tf
import numpy as np

from models.model import Model
from utils.loader import load_sentences, update_tag_scheme
from utils.loader import char_mapping, tag_mapping
from utils.loader import augment_with_pretrained, prepare_dataset
from utils.utils import get_logger, make_path, clean, create_model, save_model
from utils.utils import print_config, save_config, load_config, test_ner
from utils.data_utils import load_word2vec, create_input, input_from_line, BatchManager
import utils.opts as opts 

import code

args = opts.train_opts()

assert args.clip < 5.1, "gradient clip should't be too much"
assert 0 <= args.dropout < 1, "dropout rate between 0 and 1"
assert args.lr > 0, "learning rate must larger than zero"
assert args.optimizer in ["adam", "sgd", "adagrad"]

# config for the model
def config_model(char_to_id, tag_to_id, intent_to_id):
    config = OrderedDict()
    config["model_type"] = args.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = args.char_dim
    config["num_tags"] = len(tag_to_id)
    config["num_intents"] = len(intent_to_id)
    config["seg_dim"] = args.seg_dim
    config["lstm_dim"] = args.lstm_dim
    config["batch_size"] = args.batch_size

    config["emb_file"] = args.emb_file
    config["clip"] = args.clip
    config["dropout_keep"] = 1.0 - args.dropout
    config["optimizer"] = args.optimizer
    config["lr"] = args.lr
    config["tag_schema"] = args.tag_schema
    config["pre_emb"] = args.pre_emb
    config["zeros"] = args.zeros
    config["lower"] = args.lower
    return config

def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results, itent_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, args.result_path)

    # for line in eval_lines:
        # logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    logger.info(eval_lines[1])
    logger.info("intent accuracy score is:{:>.3f}".format(itent_results[0]))

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()

        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1

def train():
    # load data sets
    train_sentences = load_sentences(args.train_file, args.lower, args.zeros)
    dev_sentences = load_sentences(args.dev_file, args.lower, args.zeros)
    test_sentences = load_sentences(args.test_file, args.lower, args.zeros)


    # Use selected tagging scheme (IOB / IOBES)
    # 检测并维护数据集的 tag 标记
    update_tag_scheme(train_sentences, args.tag_schema)
    update_tag_scheme(test_sentences, args.tag_schema)
    update_tag_scheme(dev_sentences, args.tag_schema)

    # create maps if not exist
    # 根据数据集创建 char_to_id, id_to_char, tag_to_id, id_to_tag 字典，并储存为 pkl 文件
    if not os.path.isfile(args.map_file):
        # create dictionary for word
        if args.pre_emb:
            dico_chars_train = char_mapping(train_sentences, args.lower)[0]
            # 利用预训练嵌入集增强（扩充）字符字典，然后返回字符与位置映射关系
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                args.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(
                train_sentences, args.lower)

        # Create a dictionary and a mapping for tags
        # 获取标记与位置映射关系
        tag_to_id, id_to_tag, intent_to_id, id_to_intent = tag_mapping(
            train_sentences)

        with open(args.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag, intent_to_id, id_to_intent], f)
    else:
        with open(args.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag, intent_to_id, id_to_intent = pickle.load(f)

    # 提取句子特征
    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, intent_to_id, args.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, intent_to_id, args.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, intent_to_id, args.lower
    )

    # code.interact(local=locals())


    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    # 获取可供模型训练的单个批次数据
    train_manager = BatchManager(train_data, args.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # make path for store log and model if not exist
    make_path(args)
    if os.path.isfile(args.config_file):
        config = load_config(args.config_file)
    else:
        config = config_model(char_to_id, tag_to_id, intent_to_id)
        save_config(config, args.config_file)
    make_path(args)

    logger = get_logger(args.log_file)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # 训练集全量跑一次需要迭代的次数
    steps_per_epoch = train_manager.len_data

    with tf.Session(config=tf_config) as sess:
        # 此处模型创建为项目最核心代码
        model = create_model(sess, Model, args.ckpt_path,
                             load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss_slot = []
        loss_intent = []

        # with tf.device("/gpu:0"):
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss_slot, batch_loss_intent = model.run_step(
                    sess, True, batch)
                loss_slot.append(batch_loss_slot)
                loss_intent.append(batch_loss_intent)

                if step % args.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "INTENT loss:{:>9.6f}, "
                                "NER loss:{:>9.6f}".format(
                                    iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss_intent), np.mean(loss_slot)))
                    loss_slot = []
                    loss_intent = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
            # if i%7 == 0:
                save_model(sess, model, args.ckpt_path, logger)
        evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def evaluate_test():
    config = load_config(args.config_file)
    logger = get_logger(args.log_file)

    with open(args.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag, intent_to_id, id_to_intent = pickle.load(
            f)

    test_sentences = load_sentences(args.test_file, args.lower, args.zeros)
    update_tag_scheme(test_sentences, args.tag_schema)
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, intent_to_id, args.lower
    )
    test_manager = BatchManager(test_data, 100)

    # limit GPU memory 限制GPU的内存大小
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, args.ckpt_path,
                             load_word2vec, config, id_to_char, logger)

        evaluate(sess, model, "test", test_manager,
                 id_to_tag, logger)

def evaluate_line():
    config = load_config(args.config_file)
    logger = get_logger(args.log_file)
    # limit GPU memory 限制GPU的内存大小
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(args.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag, intent_to_id, id_to_intent = pickle.load(
            f)
    with tf.Session(config = tf_config) as sess:
        model = create_model(sess, Model, args.ckpt_path,
                             load_word2vec, config, id_to_char, logger)
        while True:
            try:
                line = input("请输入测试句子:")
                result = model.evaluate_line(
                    sess, input_from_line(line, char_to_id), id_to_tag, id_to_intent)
                print(result)
            except Exception as e:
                logger.info(e)

def main(args):
    if args.train:
        if args.clean:
            clean(args)
        print("开始训练模型！！！")
        train()
    else:
        evaluate_line()
        # evaluate_test()

if __name__ == "__main__":
    main(args)
