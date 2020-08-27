# encoding=utf-8
import os
import shutil
from src.layer.bert_softmax import BertSoftmax
from src.util.logger import setlogger
from src.util.yaml_util import loadyaml
import tensorflow as tf
from src.layer.bert_sigmoid import BertSigmoid
from src.layer.bert_textcnn import BertTextCNN


config = loadyaml('./conf/bert_toolkit_tf.yaml')
logger = setlogger(config)


def test_bert_softmax(text):
    train_path = config['train_path']
    eval_path = config['eval_path']
    bert_path = config['bert_path']
    save_path = config['save_path']
    # tf配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # bert配置
    bert_cfg = {
        'bert_path': bert_path,
        'train_path': train_path,
        'eval_path': eval_path,
        'max_length': 128,
        'batch_size': 32,
        'save_path': save_path,
        'learning_rate': 2e-5,
        'epoch': 3,
        'save_checkpoints_steps': 100,
        'tf_config': tf_config
    }
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    model = BertSoftmax(**bert_cfg)
    model.fit()
    print(model.evaluate())
    model.save('./model/bert_softmax')
    model = BertSoftmax(**bert_cfg)
    model.load('./model/bert_softmax')
    results = model.predict(text)
    print(results)


def test_bert_sigmoid(text):
    train_path = './data/atis-train1.iob'
    eval_path = './data/atis-dev1.iob'
    bert_path = config['bert_path']
    save_path = config['save_path']
    # tf配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # bert配置
    bert_cfg = {
        'bert_path': bert_path,
        'train_path': train_path,
        'eval_path': eval_path,
        'max_length': 128,
        'batch_size': 32,
        'save_path': save_path,
        'learning_rate': 2e-5,
        'epoch': 3,
        'save_checkpoints_steps': 100,
        'tf_config': tf_config
    }
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    model = BertSigmoid(**bert_cfg)
    model.fit()
    print(model.evaluate())
    model.save('./model/bert_sigmoid')
    model = BertSoftmax(**bert_cfg)
    model.load('./model/bert_sigmoid')
    results = model.predict(text)
    print(results)


def test_bert_textcnn(text):
    train_path = config['train_path']
    eval_path = config['eval_path']
    bert_path = config['bert_path']
    save_path = config['save_path']
    # tf配置
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    tf_config = tf.ConfigProto(log_device_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # bert配置
    bert_cfg = {
        'bert_path': bert_path,
        'train_path': train_path,
        'eval_path': eval_path,
        'max_length': 128,
        'batch_size': 32,
        'save_path': save_path,
        'learning_rate': 2e-5,
        'epoch': 3,
        'save_checkpoints_steps': 100,
        'tf_config': tf_config,
        'filters': 16,
        'kernel_size': 3,
        'strides': 1,
        'pool_size': 3
    }
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    model = BertTextCNN(**bert_cfg)
    model.fit()
    print(model.evaluate())
    model.save('./model/bert_textcnn')
    model = BertTextCNN(**bert_cfg)
    model.load('./model/bert_textcnn')
    results = model.predict(text)
    print(results)


if __name__ == '__main__':
    text = 'what are the flights and fares from atlanta to philadelphia'
    test_bert_softmax(text)
    test_bert_sigmoid(text)
    test_bert_textcnn(text)