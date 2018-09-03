from __future__ import print_function
import argparse
import os

root_path = os.getcwd() + os.sep

def str2bol(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def train_opts():
    parser = argparse.ArgumentParser(
        description='Learning with Bi-LSTM + CRF or IDCNN + CRF together with Language Model')
    parser.add_argument('--clean', type=str2bol, default=False,
                        help='clean train folder')
    parser.add_argument('--train', type=str2bol, default=True,
                        help='Whether train the model')
    
    # configurations for the model
    parser.add_argument('--seg_dim', type=int, default=20,
                        help='Embedding size for segmentation, 0 if not used')
    parser.add_argument('--char_dim', type=int, default=100,
                        help='Embedding size for characters')
    parser.add_argument('--lstm_dim', type=int, default=100,
                        help='Num of hidden units in LSTM, or num of filters in IDCNN')
    parser.add_argument('--tag_schema', type=str, default='iobes',
                        help='tagging schema iobes or iob')

    # configurations for training
    parser.add_argument('--clip', type=float, default=5,
                        help='Gradient clip')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer for training')

    parser.add_argument('--pre_emb', type=str2bol, default=True,
                        help='Whether use pre-trained embedding')
    parser.add_argument('--zeros', type=str2bol, default=True,
                        help='Whether replace digits with zero')
    parser.add_argument('--lower', type=str2bol, default=False,
                        help='Whether lower case')

    parser.add_argument('--max_epoch', type=int, default=100,
                        help='maximum training epochs')
    parser.add_argument('--steps_check', type=int, default=100,
                        help='steps per checkpoint')

    parser.add_argument('--ckpt_path', type=str, default='ckpt',
                        help='Path to save model')
    parser.add_argument('--summary_path', type=str, default='summary',
                        help='Path to store summaries')
    parser.add_argument('--result_path', type=str, default='result',
                        help='Path for results')
    parser.add_argument('--configs_path', type=str, default='configs',
                        help='Path for configs')

    parser.add_argument('--log_file', type=str,
                        default=os.path.join(root_path + "log", "train.log"),
                        help='File for log')
    parser.add_argument('--config_file', type=str,
                        default=os.path.join(root_path + "configs", "config_file"),
                        help='File for config')
    parser.add_argument('--map_file', type=str,
                        default=os.path.join(root_path + "configs", "maps.pkl"),
                        help='File for maps')
    parser.add_argument('--vocab_file', type=str,
                        default=os.path.join(root_path + "configs", "vocab.json"),
                        help='File for vocab')
    parser.add_argument('--emb_file', type=str,
                        default=os.path.join(
                            root_path + "assets" + os.sep + "cooked_corpus", "vec.txt"),
                        help='File for pre_trained embedding')
    parser.add_argument('--train_file', type=str,
                        default=os.path.join(
                            root_path + "assets" + os.sep + "cooked_corpus", "example.train"),
                        help='File for train data')
    parser.add_argument('--dev_file', type=str,
                        default=os.path.join(
                            root_path + "assets" + os.sep + "cooked_corpus", "example.dev"),
                        help='File for dev data')
    parser.add_argument('--test_file', type=str,
                        default=os.path.join(
                            root_path + "assets" + os.sep + "cooked_corpus", "example.test"),
                        help='File for test data')

    # model type, idcnn or bilstm
    parser.add_argument('--model_type', type=str,
                        default='idcnn',
                        help='Model type, can be idcnn or bilstm')
    # parser.add_argument('--model_type', type=str,
    #                     default='bilstm',
    #                     help='Model type, can be idcnn or bilstm')
    
    return parser.parse_args()
