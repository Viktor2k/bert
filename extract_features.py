# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re
import os
import time
import csv

import modeling
import tokenization
import tensorflow as tf


from itertools import count

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None, "If using a TPU, the address of the master.")

flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
        "use_one_hot_embeddings", False,
        "If True, tf.one_hot will be used for embedding lookups, otherwise "
        "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
        "since it is much faster.")


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, label):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputSubexample(object):

    _ids = count(0)

    def __init__(self, example_id, seq_id, tokens_a, embedding_mask):
        self.unique_id = next(self._ids)
        self.seq_coord = (example_id, seq_id)
        self.tokens_a = tokens_a
        self.embedding_mask = embedding_mask


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, label, tokens, input_ids, input_mask, input_type_ids, embedding_mask, seq_coord):
        self.unique_id = unique_id
        self.label = label
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.embedding_mask = embedding_mask
        self.seq_coord = seq_coord


def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []

    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({"unique_ids": tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                                                "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
                                                "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
                                                "input_type_ids": tf.constant(all_input_type_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indexes, use_tpu, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(config=bert_config, is_training=False, input_ids=input_ids, input_mask=input_mask, token_type_ids=input_type_ids, use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape, init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {"unique_id": unique_ids, }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def _generate_subexamples(example, seq_length, tokenizer):
    '''Takes an example and returns a list of subexamples that fit the size of seq_length'''
    #logger.debug('------- New row to process -------')
    window_size = int(seq_length / 2)
    context = int(window_size / 2) - 1    # To make room for [CLS] and [SEP] tokens.
    stride = window_size
    tokens = tokenizer.tokenize(example.text_a)
    n_tokens = len(tokens)

    subexamples = []

    if n_tokens < window_size + 2 * context:    # The entire text will fit within the scope of window and context in the first pass. No need to process this text twice.
        subexamples.append(InputSubexample(example.unique_id, 0, tokens, [1] * n_tokens))
        return subexamples
    #end if

    for i, embedding_start in enumerate(range(0, n_tokens, stride)):
        embedding_mask = [0] * (window_size + 2 * context)    # Matrix for keeping track of what parts of each sequence is supposed to be embedded at each step.
        if embedding_start == 0:    # first tokens
            start = 0
            end = window_size + 2 * context
            embedding_mask[embedding_start:window_size] = [1] * (window_size - embedding_start)
            assert(len(embedding_mask) == (window_size + 2 * context))

        elif n_tokens - embedding_start < window_size:    # embedding-window overlapping end of string.
            start = n_tokens - (window_size + 2 * context)
            end = n_tokens
            embedding_mask[embedding_start - start:end - start] = [1] * (end - start - (embedding_start - start))
            assert(len(embedding_mask) == (window_size + 2 * context))

        else:    # somewhere in the middle
            end = min(embedding_start + window_size + context, n_tokens)    # Prevents context from falling outside range of text
            start = end - (window_size + 2 * context)    # Start is always fixed size from end. This makes sure the number of words in the context stays fixed.
            embedding_mask[embedding_start - start:embedding_start + window_size - start] = [1] * ((embedding_start + window_size - start) - (embedding_start - start))
            assert(len(embedding_mask) == (window_size + 2 * context))
        #end if

        subexamples.append(InputSubexample(example.unique_id, i, tokens[start:end], embedding_mask))
    # end for

    #extract the tokens that should be embedded in each step and make sure they add up to be the entire text.
    tokens_to_be_embedded = []
    for subexample in subexamples:
        for token, embedding_bool in zip(subexample.tokens_a, subexample.embedding_mask):
            if embedding_bool:
                tokens_to_be_embedded.append(token)
    assert(tokens_to_be_embedded == tokens)

    return subexamples


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    #TODO: Reshape examples so that they fit the seq_length

    features = []
    for (ex_index, example) in enumerate(examples):
        subexamples = _generate_subexamples(example, seq_length, tokenizer)

        for subexample in subexamples:
            tokens_a = subexample.tokens_a
            # The convention in BERT is:
            # (a) For sequence pairs:
            #    tokens:     [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #    type_ids: 0         0    0        0        0         0             0 0         1    1    1    1     1 1
            # (b) For single sequences:
            #    tokens:     [CLS] the dog is hairy . [SEP]
            #    type_ids: 0         0     0     0    0         0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            embedding_mask = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            embedding_mask.append(0)    # I dont need the embedding for the [CLS]-token
            for token, embedding_bool in zip(tokens_a, subexample.embedding_mask):
                tokens.append(token)
                embedding_mask.append(embedding_bool)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)
            embedding_mask.append(0)    # I dont need the embedding for the [SEP]-token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)
                embedding_mask.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length
            assert len(embedding_mask) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                #tf.logging.info("unique_id: %d_%d" % subexample.unique_id)
                tf.logging.info("unique_id: %s" % (subexample.unique_id))
                tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
                tf.logging.info("embedding_mask: %s" % " ".join([str(x) for x in embedding_mask]))

            features.append(InputFeatures(unique_id=subexample.unique_id, label=example.label, tokens=tokens, input_ids=input_ids, input_mask=input_mask, input_type_ids=input_type_ids, embedding_mask=embedding_mask, seq_coord=subexample.seq_coord))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0

    with tf.gfile.GFile(input_file, "r") as reader:
        csv_reader = csv.reader(reader, delimiter=';')
        while True:
            try:
                csv_row = next(csv_reader)
            except StopIteration:
                break
            #end try

            label = csv_row[0]
            line = csv_row[1]

            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b, label=label))
            unique_id += 1
    return examples


def _write_features_to_file(writer, doc_features, doc_num, label):
    temp_features = []
    for key in sorted(doc_features.keys()):
        temp_features.extend(doc_features[key])
    #end for
    writer.write(json.dumps({"linex_index": doc_num, "label": label, "features": temp_features}) + "\n")
    #end with
#end def


def combine_features(fractured_json, combined_json):
    with codecs.getwriter("utf-8")(tf.gfile.Open(combined_json, "w")) as writer, tf.gfile.GFile(fractured_json, "r") as f:
        row_data = None
        while True:
            if not row_data:
                try:
                    row_data = json.loads(next(f))
                except:
                    break
                #end try
            #end if

            doc_features = {}
            # Extract the first row, or the first row with this sequence
            doc_num = row_data['sequence_coordinate'][0]
            seq_num = row_data['sequence_coordinate'][1]
            doc_features[seq_num] = row_data['features']
            label = row_data['label']
            #print(f'Working with document {(doc_num, seq_num)}')
            while True:
                try:
                    #Extract the following row and test its doc_num
                    row_data = json.loads(next(f))
                    d = row_data["sequence_coordinate"][0]
                    r = row_data["sequence_coordinate"][1]
                    #print(f'Working with document {(d, r)}')
                    if row_data['sequence_coordinate'][0] == doc_num:  # same doc_num as before, extract features and continue
                        doc_features[row_data['sequence_coordinate'][1]] = row_data['features']
                    else:  # i
                        _write_features_to_file(writer, doc_features, doc_num, label)
                        #print(f'Wrote {doc_num} to feature list from try')
                        break
                        # append features from doc_features to feature list according to their key-value
                except:
                    _write_features_to_file(writer, doc_features, doc_num, label)
                    #print(f'Wrote {doc_num} to feature list from except')
                    row_data = None
                    break
                #end try
            #end while
        #end while
    #end with
#end def


def sort_input(input_file, output_file):
    input_data = []
    idx_to_n = {}
    with tf.gfile.Open(input_file) as f:
        for idx, row in enumerate(f):
            input_data.append(row)

            # Heuristic for number of sentences in this document
            idx_to_n[idx] = row.count('.') + row.count('!') + row.count('?')
        #end for
    #end with

    sort_order = sorted(idx_to_n.items(), key=lambda kv: kv[1])  # list of tuples (idx, n_sent)
    with codecs.getwriter("utf-8")(tf.gfile.Open(output_file, 'w')) as f:
        for s in sort_order:
            input_data[s[0]]
        #end for
    #end with
#end def


def batch_input(input_file, batch_size=1000):
    '''returns list names for the batched file'''

    path, file_name = os.path.split(input_file)
    file_, ext = os.path.splitext(file_name)  # <dataset>_<datatype>.csv
    
    cur_batch = []
    batch_files = []
    with tf.gfile.Open(input_file) as input_f:
        csv_reader = csv.reader(input_f, delimiter=';')
        for i, row in enumerate(csv_reader, start=1):
            cur_batch.append(row)
            if i % batch_size == 0:
                batch_file_name = f'{os.path.join(path, file_)}_batch_{int(i/batch_size)}{ext}'  # <dataset>_<datatype>_batch_<batch_id>.csv
                batch_files.append(batch_file_name)
                with codecs.getwriter("utf-8")(tf.gfile.Open(batch_file_name, "w")) as output_f:
                    writer = csv.writer(output_f, delimiter=';')
                    for b in cur_batch:
                        writer.writerow(b)
                    #end for
                #end with
                cur_batch = []
            #end if
        #end for

        # write the remaining files to a unique batch
        batch_file_name = f'{os.path.join(path, file_)}_batch_0{ext}'
        batch_files.append(batch_file_name)

        with codecs.getwriter("utf-8")(tf.gfile.Open(batch_file_name, "w")) as output_f:
            writer = csv.writer(output_f, delimiter=';')
            for b in cur_batch:
                writer.writerow(b)
            #end for
        #end with

    #end with

    return batch_files
#end def


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(master=FLAGS.master, tpu_config=tf.contrib.tpu.TPUConfig(num_shards=FLAGS.num_tpu_cores, per_host_input_for_training=is_per_host))

    # Run batch data and then loop over these files
    tf.logging.info('Batching input file')
    batch_files = batch_input(FLAGS.input_file)
    tf.logging.info(f'Done with batching input file. Total batches: {len(batch_files)}')
    model_fn = model_fn_builder(bert_config=bert_config, init_checkpoint=FLAGS.init_checkpoint, layer_indexes=layer_indexes, use_tpu=FLAGS.use_tpu, use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(use_tpu=FLAGS.use_tpu, model_fn=model_fn, config=run_config, predict_batch_size=FLAGS.batch_size, train_batch_size=256)

    output_batch_files = []
    for b, input_file in enumerate(batch_files, start=1):
        start = time.time()
        examples = read_examples(input_file)
        features = convert_examples_to_features(examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature
        #end for

        input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

        # BERT outputs tokens for each subexample per row. These will later be combined to a single row for each instance, but saved in the temp file in the mean-while. 
        path, _ = os.path.split(input_file)
        temp_file_name = 'temp_batch_storage.json'
        temp_output_file = os.path.join(path, temp_file_name)

        with codecs.getwriter("utf-8")(tf.gfile.Open(temp_output_file, "w")) as writer:
            for result in estimator.predict(input_fn, yield_single_examples=True):
                unique_id = int(result["unique_id"])
                feature = unique_id_to_feature[unique_id]
                output_json = collections.OrderedDict()
                output_json["linex_index"] = unique_id
                output_json["sequence_coordinate"] = feature.seq_coord
                output_json["label"] = feature.label
                all_features = []
                for (i, token) in enumerate(feature.tokens):
                    if not feature.embedding_mask[i]:
                        continue
                    #end if
                    all_layers = []
                    for (j, layer_index) in enumerate(layer_indexes):
                        layer_output = result["layer_output_%d" % j]
                        layers = collections.OrderedDict()
                        layers["index"] = layer_index
                        layers["values"] = [round(float(x), 6) for x in layer_output[i:(i + 1)].flat]
                        all_layers.append(layers)
                    #end for

                    features = collections.OrderedDict()
                    features["token"] = token
                    features["layers"] = all_layers
                    all_features.append(features)
                #end for

                output_json["features"] = all_features
                writer.write(json.dumps(output_json) + "\n")
            #end for
        #end with
        output_file = f'{os.path.splitext(input_file)[0]}.json'
        output_batch_files.append(output_file)
        combine_features(temp_output_file, output_file)
        tf.logging.info(f"\n\nSaved batch {b} of {len(batch_files)} to {output_batch_files} which took {round(time.time()-start)} seconds.\n\n")
    #end for


#end def

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    tf.app.run()
