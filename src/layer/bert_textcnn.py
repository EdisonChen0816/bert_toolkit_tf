# encoding=utf-8
import os
import glob
import collections
import tensorflow as tf
from src.bert import modeling, optimization, tokenization
from src.bert.modeling import InputFeatures, Processor
import pickle


def create_model(bert_config, is_training, input_ids, input_mask, num_labels, filters, kernel_size,
                 strides, pool_size, labels=None):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        use_one_hot_embeddings=False)
    bert_embedding = model.get_all_encoder_layers()[-1]
    _, max_len, dim = bert_embedding.shape
    x_input = tf.reshape(bert_embedding, shape=[-1, max_len, dim, 1])
    with tf.variable_scope("loss"):
        if is_training:
            dropout = 0.1
        else:
            dropout = 0
        conv = tf.layers.conv2d(x_input, filters, kernel_size=(kernel_size, dim), strides=strides,
                                activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(conv, pool_size=(pool_size, 1), strides=strides)
        fc1 = tf.layers.flatten(pool, name="fc1")
        fc2 = tf.layers.dense(fc1, 128)
        fc2 = tf.layers.dropout(fc2, rate=dropout)
        out = tf.layers.dense(fc2, num_labels)
        probabilities = tf.nn.softmax(out, axis=-1)
        if labels is None:
            return probabilities
        log_probs = tf.nn.log_softmax(out, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return loss, per_example_loss, out, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, filters, kernel_size, strides,
                     pool_size, learning_rate=5e-5, num_train_steps=0, num_warmup_steps=0):
    def model_fn(features, mode):
        input_ids, input_mask, label_ids = [features.get(k) for k in \
                                            ("input_ids", "input_mask", "label_ids")]
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, num_labels, filters, kernel_size, strides, pool_size, label_ids)
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            accu = tf.metrics.accuracy(labels=label_ids, predictions= \
                tf.argmax(logits, axis=-1, output_type=tf.int32))
            loss = tf.metrics.mean(values=per_example_loss)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss,
                                                     eval_metric_ops={"eval_accu": accu, "eval_loss": loss})
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"prob": probabilities})
        return output_spec
    return model_fn


def convert_single_example(ex_index, example, label_list, max_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens = tokenizer.tokenize(example.text)
    tokens = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(tokens))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("label: %s(id=%d)" % (example.label, label_id))
    return InputFeatures(input_ids=input_ids, input_mask=input_mask, label_id=label_id)


def file_based_convert_examples_to_features(examples, label_list, max_length, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for ex_index, example in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list, max_length, tokenizer)

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature([feature.label_id])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch_size):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64)}

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn():
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat().shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
        return d
    return input_fn


def dump_model_fn_builder(bert_config, num_labels, init_checkpoint, filters, kernel_size, strides, pool_size):
    def model_fn(features, mode):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        proba = create_model(bert_config, False, input_ids, input_mask, num_labels, filters, kernel_size, strides, pool_size)
        tvars = tf.trainable_variables()
        assignment_map, _ = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        export_outputs = {
            'predict': tf.estimator.export.PredictOutput(proba)
        }
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=proba, export_outputs=export_outputs)
        return output_spec
    return model_fn


def serving_input_receiver_fn(max_length):
    input_ids = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=[None, max_length], dtype=tf.int32, name="input_mask")
    features = {"input_ids": input_ids, "input_mask": input_mask}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features)


class BertTextCNN:

    def __init__(self, bert_path, train_path, eval_path, max_length, batch_size,
                 save_path, learning_rate, epoch, save_checkpoints_steps, tf_config,
                 filters, kernel_size, strides, pool_size, do_lower_case=True, warmup_ratio=.1):
        self.bert_path = bert_path
        self.train_path = train_path
        self.eval_path = eval_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.save_checkpoints_steps = save_checkpoints_steps
        self.tf_config = tf_config
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.pool_size = pool_size
        self.do_lower_case = do_lower_case
        self.warmup_ratio = warmup_ratio
        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.save_config()

    def save_config(self):
        self.config = {}
        self.config['bert_path'] = self.bert_path
        self.config['train_path'] = self.train_path
        self.config['eval_path'] = self.eval_path
        self.config['max_length'] = self.max_length
        self.config['batch_size'] = self.batch_size
        self.config['save_path'] = self.save_path
        self.config['learning_rate'] = self.learning_rate
        self.config['epoch'] = self.epoch
        self.config['save_checkpoints_steps'] = self.save_checkpoints_steps
        self.config['do_lower_case'] = self.do_lower_case
        self.config['warmup_ratio'] = self.warmup_ratio
        self.config['filters'] = self.filters
        self.config['kernel_size'] = self.kernel_size
        self.config['strides'] = self.strides
        self.config['pool_size'] = self.pool_size

    def fit(self):
        tf.gfile.MakeDirs(self.save_path)
        processor = Processor()
        train_examples, labels = processor.get_train_examples(self.train_path)
        num_train_steps = int(len(train_examples) / self.batch_size * self.epoch)
        num_warmup_steps = 0 if self.model else int(num_train_steps * self.warmup_ratio)
        if not self.model:
            self.labels = labels
            self.config['labels'] = self.labels
            init_checkpoint = os.path.join(self.bert_path, "bert_model.ckpt")
            bert_config_file = os.path.join(self.bert_path, "bert_config.json")
            bert_config = modeling.BertConfig.from_json_file(bert_config_file)
            model_fn = model_fn_builder(
                bert_config=bert_config,
                num_labels=len(labels),
                init_checkpoint=init_checkpoint,
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                pool_size=self.pool_size,
                learning_rate=self.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)
            run_config = tf.estimator.RunConfig(
                model_dir=self.save_path,
                save_checkpoints_steps=self.save_checkpoints_steps,
                session_config=self.tf_config)
            self.model = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config)
        vocab_file = os.path.join(self.bert_path, "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=self.do_lower_case)
        train_file = os.path.join(self.save_path, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, labels, self.max_length, self.tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_length,
            is_training=True,
            drop_remainder=True,
            batch_size=self.batch_size)
        self.model.train(input_fn=train_input_fn, max_steps=num_train_steps)
        with open(os.path.join(self.save_path, "config"), "wb") as out:
            pickle.dump(self.config, out)

    def evaluate(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        processor = Processor()
        eval_examples, _ = processor.get_test_examples(self.eval_path)
        eval_file = os.path.join(self.save_path, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, self.labels, self.max_length, self.tokenizer, eval_file)
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_length,
            is_training=False,
            drop_remainder=False,
            batch_size=self.batch_size)
        return self.model.evaluate(input_fn=eval_input_fn, steps=None)

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens[:self.max_length - 2] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_ids += [0] * (self.max_length - len(input_ids))
        input_mask += [0] * (self.max_length - len(input_mask))
        features = {"input_ids": [input_ids], "input_mask": [input_mask]}
        probs = self.predictor(features)["output"].tolist()[0]
        return [{"label": name, "prob": prob} for name, prob in zip(self.config['labels'], probs)]

    def load(self, dir):
        assert os.path.exists(dir)
        with open(os.path.join(dir, "config"), "rb") as fin:
            self.config = pickle.load(fin)
        vocab_file = os.path.join(self.config['bert_path'], "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file,
            do_lower_case=self.config['do_lower_case'])
        saved_model = sorted(glob.glob(os.path.join(dir, "exported", "*")))[-1]
        self.predictor = tf.contrib.predictor.from_saved_model(saved_model)

    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, "config"), "wb") as out:
            pickle.dump(self.config, out)
        bert_config_file = os.path.join(self.bert_path, "bert_config.json")
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        predictor = tf.estimator.Estimator(
            model_fn=dump_model_fn_builder(
                bert_config=bert_config,
                num_labels=len(self.labels),
                init_checkpoint=self.save_path,
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                pool_size=self.pool_size
            ),
            config=tf.estimator.RunConfig(model_dir=self.save_path))
        predictor.export_savedmodel(os.path.join(dir, "exported"),
                                    serving_input_receiver_fn(self.max_length))


if __name__ == "__main__":
    pass