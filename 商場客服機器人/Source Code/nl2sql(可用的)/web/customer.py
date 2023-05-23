import os
import re
import json
import math
import numpy as np
from tqdm import tqdm_notebook as tqdm

from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths

import keras.backend as K
from keras.layers import Input, Dense, Lambda, Multiply, Masking, Concatenate
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import Sequence

from nl2sql.utils import read_data, read_tables, SQL, MultiSentenceTokenizer, Query, Question, Table
from tensorflow import keras
from keras_radam import RAdam

import random
from collections import defaultdict

import cn2an
from tensorflow.keras.optimizers import Adam
            
import tensorflow as tf


def remove_brackets(s):
    '''
    Remove brackets [] () from text
    '''
    return re.sub(r'[\(\（].*[\)\）]', '', s)

class QueryTokenizer(MultiSentenceTokenizer):
    """
    Tokenize query (question + table header) and encode to integer sequence.
    Using reserved tokens [unused11] and [unused12] for classification
    """
    
    col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}
    
    def tokenize(self, query: Query, col_orders=None):
        """
        Tokenize quesiton and columns and concatenate.
        
        Parameters:
        query (Query): A query object contains question and table
        col_orders (list or numpy.array): For re-ordering the header columns
        
        Returns:
        token_idss: token ids for bert encoder
        segment_ids: segment ids for bert encoder
        header_ids: positions of columns
        """
        
        question_tokens = [self._token_cls] + self._tokenize(query.question.text)
        header_tokens = []
        
        if col_orders is None:
            col_orders = np.arange(len(query.table.header))
        
        header = [query.table.header[i] for i in col_orders]
        
        for col_name, col_type in header:
            col_type_token = self.col_type_token_dict[col_type]
            col_name = remove_brackets(col_name)
            col_name_tokens = self._tokenize(col_name)
            col_tokens = [col_type_token] + col_name_tokens
            header_tokens.append(col_tokens)
            
        all_tokens = [question_tokens] + header_tokens
        return self._pack(*all_tokens)
    
    def encode(self, query:Query, col_orders=None):
        tokens, tokens_lens = self.tokenize(query, col_orders)
        token_ids = self._convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        header_indices = np.cumsum(tokens_lens)
        return token_ids, segment_ids, header_indices[:-1]

class SqlLabelEncoder:
    """
    Convert SQL object into training labels.
    """
    def encode(self, sql: SQL, num_cols):
        cond_conn_op_label = sql.cond_conn_op
        
        sel_agg_label = np.ones(num_cols, dtype='int32') * len(SQL.agg_sql_dict)
        for col_id, agg_op in zip(sql.sel, sql.agg):
            if col_id < num_cols:
                sel_agg_label[col_id] = agg_op
            
        cond_op_label = np.ones(num_cols, dtype='int32') * len(SQL.op_sql_dict)
        for col_id, cond_op, _ in sql.conds:
            if col_id < num_cols:
                cond_op_label[col_id] = cond_op
            
        return cond_conn_op_label, sel_agg_label, cond_op_label
    
    def decode(self, cond_conn_op_label, sel_agg_label, cond_op_label):
        cond_conn_op = int(cond_conn_op_label)
        sel, agg, conds = [], [], []

        for col_id, (agg_op, cond_op) in enumerate(zip(sel_agg_label, cond_op_label)):
            if agg_op < len(SQL.agg_sql_dict):
                sel.append(col_id)
                agg.append(int(agg_op))
            if cond_op < len(SQL.op_sql_dict):
                conds.append([col_id, int(cond_op)])
        return {
            'sel': sel,
            'agg': agg,
            'cond_conn_op': cond_conn_op,
            'conds': conds
        }    

class DataSequence(Sequence):
    """
    Generate training data in batches
    
    """
    def __init__(self, 
                 data, 
                 tokenizer, 
                 label_encoder, 
                 is_train=True, 
                 max_len=160, 
                 batch_size=32, 
                 shuffle=True, 
                 shuffle_header=True, 
                 global_indices=None):
        
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.shuffle = shuffle
        self.shuffle_header = shuffle_header
        self.is_train = is_train
        self.max_len = max_len
        
        if global_indices is None:
            self._global_indices = np.arange(len(data))
        else:
            self._global_indices = global_indices

        if shuffle:
            np.random.shuffle(self._global_indices)
    
    def _pad_sequences(self, seqs, max_len=None):
        padded = pad_sequences(seqs, maxlen=None, padding='post', truncating='post')
        if max_len is not None:
            padded = padded[:, :max_len]
        return padded
    
    def __getitem__(self, batch_id):
        batch_data_indices = \
            self._global_indices[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_data_indices]
        
        TOKEN_IDS, SEGMENT_IDS = [], []
        HEADER_IDS, HEADER_MASK = [], []
        
        COND_CONN_OP = []
        SEL_AGG = []
        COND_OP = []
        
        for query in batch_data:
            question = query.question.text
            table = query.table
            
            col_orders = np.arange(len(table.header))
            if self.shuffle_header:
                np.random.shuffle(col_orders)
            
            token_ids, segment_ids, header_ids = self.tokenizer.encode(query, col_orders)
            header_ids = [hid for hid in header_ids if hid < self.max_len]
            header_mask = [1] * len(header_ids)
            col_orders = col_orders[: len(header_ids)]
            
            TOKEN_IDS.append(token_ids)
            SEGMENT_IDS.append(segment_ids)
            HEADER_IDS.append(header_ids)
            HEADER_MASK.append(header_mask)
            
            if not self.is_train:
                continue
            sql = query.sql
            
            cond_conn_op, sel_agg, cond_op = self.label_encoder.encode(sql, num_cols=len(table.header))
            
            sel_agg = sel_agg[col_orders]
            cond_op = cond_op[col_orders]
            
            COND_CONN_OP.append(cond_conn_op)
            SEL_AGG.append(sel_agg)
            COND_OP.append(cond_op)
            
        TOKEN_IDS = self._pad_sequences(TOKEN_IDS, max_len=self.max_len)
        SEGMENT_IDS = self._pad_sequences(SEGMENT_IDS, max_len=self.max_len)
        HEADER_IDS = self._pad_sequences(HEADER_IDS)
        HEADER_MASK = self._pad_sequences(HEADER_MASK)
        
        inputs = {
            'input_token_ids': TOKEN_IDS,
            'input_segment_ids': SEGMENT_IDS,
            'input_header_ids': HEADER_IDS,
            'input_header_mask': HEADER_MASK
        }
        
        if self.is_train:
            SEL_AGG = self._pad_sequences(SEL_AGG)
            SEL_AGG = np.expand_dims(SEL_AGG, axis=-1)
            COND_CONN_OP = np.expand_dims(COND_CONN_OP, axis=-1)
            COND_OP = self._pad_sequences(COND_OP)
            COND_OP = np.expand_dims(COND_OP, axis=-1)

            outputs = {
                'output_sel_agg': SEL_AGG,
                'output_cond_conn_op': COND_CONN_OP,
                'output_cond_op': COND_OP
            }
            return inputs, outputs
        else:
            return inputs
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._global_indices)

def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    return tf.compat.v1.batch_gather(seq, idxs)

def outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens, label_encoder):
    """
    Generate sqls from model outputs
    """
    preds_cond_conn_op = np.argmax(preds_cond_conn_op, axis=-1)
    preds_cond_op = np.argmax(preds_cond_op, axis=-1)

    sqls = []
    
    for cond_conn_op, sel_agg, cond_op, header_len in zip(preds_cond_conn_op, 
                                                          preds_sel_agg, 
                                                          preds_cond_op, 
                                                          header_lens):
        sel_agg = sel_agg[:header_len]
        # force to select at least one column for agg
        sel_agg[sel_agg == sel_agg[:, :-1].max()] = 1
        sel_agg = np.argmax(sel_agg, axis=-1)
        
        sql = label_encoder.decode(cond_conn_op, sel_agg, cond_op)
        sql['conds'] = [cond for cond in sql['conds'] if cond[0] < header_len]
        
        sel = []
        agg = []
        for col_id, agg_op in zip(sql['sel'], sql['agg']):
            if col_id < header_len:
                sel.append(col_id)
                agg.append(agg_op)
                
        sql['sel'] = sel
        sql['agg'] = agg
        sqls.append(sql)
    return sqls

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def cn_to_an(string):
    try:
        return str(cn2an.cn2an(string, 'normal'))
    except ValueError:
        return string

def an_to_cn(string):
    try:
        return str(cn2an.an2cn(string))
    except ValueError:
        return string

def str_to_num(string):
    try:
        float_val = float(cn_to_an(string))
        if int(float_val) == float_val:   
            return str(int(float_val))
        else:
            return str(float_val)
    except ValueError:
        return None

def str_to_year(string):
    year = string.replace('年', '')
    year = cn_to_an(year)
    if is_float(year) and float(year) < 1900:
        year = int(year) + 2000
        return str(year)
    else:
        return None
    
def load_json(json_file):
    result = []
    if json_file:
        with open(json_file) as file:
            for line in file:
                result.append(json.loads(line))
    return result

class QuestionCondPair:
    def __init__(self, query_id, question, cond_text, cond_sql, label):
        self.query_id = query_id
        self.question = question
        self.cond_text = cond_text
        self.cond_sql = cond_sql
        self.label = label

    def __repr__(self):
        repr_str = ''
        repr_str += 'query_id: {}\n'.format(self.query_id)
        repr_str += 'question: {}\n'.format(self.question)
        repr_str += 'cond_text: {}\n'.format(self.cond_text)
        repr_str += 'cond_sql: {}\n'.format(self.cond_sql)
        repr_str += 'label: {}\n'.format(self.label)
        return repr_str

    
class NegativeSampler:
    """
    从 question - cond pairs 中采样
    """
    def __init__(self, neg_sample_ratio=10):
        self.neg_sample_ratio = neg_sample_ratio
    
    def sample(self, data):
        positive_data = [d for d in data if d.label == 1]
        negative_data = [d for d in data if d.label == 0]
        negative_sample = random.sample(negative_data, 
                                        len(positive_data) * self.neg_sample_ratio)
        return positive_data + negative_sample

    
class FullSampler:
    """
    不抽样，返回所有的 pairs
    
    """
    def sample(self, data):
        return data

class CandidateCondsExtractor:
    """
    params:
        - share_candidates: 在同 table 同 column 中共享 real 型 candidates
    """
    CN_NUM = '〇一二三四五六七八九零壹贰叁肆伍陆柒捌玖貮两'
    CN_UNIT = '十拾百佰千仟万萬亿億兆点'
    
    def __init__(self, share_candidates=True):
        self.share_candidates = share_candidates
        self._cached = False
    
    def build_candidate_cache(self, queries):
        self.cache = defaultdict(set)
        print('building candidate cache')
        for query_id, query in tqdm(enumerate(queries), total=len(queries)):
            value_in_question = self.extract_values_from_text(query.question.text)
            
            for col_id, (col_name, col_type) in enumerate(query.table.header):
                value_in_column = self.extract_values_from_column(query, col_id)
                if col_type == 'text':
                    cond_values = value_in_column
                elif col_type == 'real':
                    if len(value_in_column) == 1: 
                        cond_values = value_in_column + value_in_question
                    else:
                        cond_values = value_in_question
                cache_key = self.get_cache_key(query_id, query, col_id)
                self.cache[cache_key].update(cond_values)
        self._cached = True
    
    def get_cache_key(self, query_id, query, col_id):
        if self.share_candidates:
            return (query.table.id, col_id)
        else:
            return (query_id, query.table.id, col_id)
        
    def extract_year_from_text(self, text):
        values = []
        num_year_texts = re.findall(r'[0-9][0-9]年', text)
        values += ['20{}'.format(text[:-1]) for text in num_year_texts]
        cn_year_texts = re.findall(r'[{}][{}]年'.format(self.CN_NUM, self.CN_NUM), text)
        cn_year_values = [str_to_year(text) for text in cn_year_texts]
        values += [value for value in cn_year_values if value is not None]
        return values
    
    def extract_num_from_text(self, text):
        values = []
        num_values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', text)
        values += num_values
        
        cn_num_unit = self.CN_NUM + self.CN_UNIT
        cn_num_texts = re.findall(r'[{}]*\.?[{}]+'.format(cn_num_unit, cn_num_unit), text)
        cn_num_values = [str_to_num(text) for text in cn_num_texts]
        values += [value for value in cn_num_values if value is not None]
    
        cn_num_mix = re.findall(r'[0-9]*\.?[{}]+'.format(self.CN_UNIT), text)
        for word in cn_num_mix:
            num = re.findall(r'[-+]?[0-9]*\.?[0-9]+', word)
            for n in num:
                word = word.replace(n, an_to_cn(n))
            str_num = str_to_num(word)
            if str_num is not None:
                values.append(str_num)
        return values
    
    def extract_values_from_text(self, text):
        values = []
        values += self.extract_year_from_text(text)
        values += self.extract_num_from_text(text)
        return list(set(values))
   
    def extract_values_from_column(self, query, col_ids):
        question = query.question.text
        question_chars = set(query.question.text)
        unique_col_values = set(query.table.df.iloc[:, col_ids].astype(str))
        select_col_values = [v for v in unique_col_values 
                             if (question_chars & set(v))]
        return select_col_values
    
    
class QuestionCondPairsDataset:
    """
    question - cond pairs 数据集
    """
    OP_PATTERN = {
        'real':
        [
            {'cond_op_idx': 0, 'pattern': '{col_name}大于{value}'},
            {'cond_op_idx': 1, 'pattern': '{col_name}小于{value}'},
            {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
        ],
        'text':
        [
            {'cond_op_idx': 2, 'pattern': '{col_name}是{value}'}
        ]
    }    
    
    def __init__(self, queries, candidate_extractor, has_label=True, model_1_outputs=None):
        self.candidate_extractor = candidate_extractor
        self.has_label = has_label
        self.model_1_outputs = model_1_outputs
        self.data = self.build_dataset(queries)
        
    def build_dataset(self, queries):
        if not self.candidate_extractor._cached:
            self.candidate_extractor.build_candidate_cache(queries)
            
        pair_data = []
        for query_id, query in enumerate(queries):
            select_col_id = self.get_select_col_id(query_id, query)
            for col_id, (col_name, col_type) in enumerate(query.table.header):
                if col_id not in select_col_id:
                    continue
                    
                cache_key = self.candidate_extractor.get_cache_key(query_id, query, col_id)
                values = self.candidate_extractor.cache.get(cache_key, [])
                pattern = self.OP_PATTERN.get(col_type, [])
                pairs = self.generate_pairs(query_id, query, col_id, col_name, 
                                               values, pattern)
                pair_data += pairs
        return pair_data
    
    def get_select_col_id(self, query_id, query):
        if self.model_1_outputs:
            select_col_id = [cond_col for cond_col, *_ in self.model_1_outputs[query_id]['conds']]
        elif self.has_label:
            select_col_id = [cond_col for cond_col, *_ in query.sql.conds]
        else:
            select_col_id = list(range(len(query.table.header)))
        return select_col_id
            
    def generate_pairs(self, query_id, query, col_id, col_name, values, op_patterns):
        pairs = []
        for value in values:
            for op_pattern in op_patterns:
                cond = op_pattern['pattern'].format(col_name=col_name, value=value)
                cond_sql = (col_id, op_pattern['cond_op_idx'], value)
                real_sql = {}
                if self.has_label:
                    real_sql = {tuple(c) for c in query.sql.conds}
                label = 1 if cond_sql in real_sql else 0
                pair = QuestionCondPair(query_id, query.question.text,
                                        cond, cond_sql, label)
                pairs.append(pair)
        return pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]    

class SimpleTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R
            
def construct_model(paths):
    token_dict = load_vocabulary(paths.vocab)
    tokenizer = SimpleTokenizer(token_dict)

    bert_model = load_trained_model_from_checkpoint(
        paths.config, paths.checkpoint, seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,), name='input_x1', dtype='int32')
    x2_in = Input(shape=(None,), name='input_x2')
    x = bert_model([x1_in, x2_in])
    x_cls = Lambda(lambda x: x[:, 0])(x)
    y_pred = Dense(1, activation='sigmoid', name='output_similarity')(x_cls)

    model = Model([x1_in, x2_in], y_pred)

    model.compile(loss={'output_similarity': 'binary_crossentropy'},
                  optimizer=Adam(1e-5),
                  metrics={'output_similarity': 'accuracy'})

    return model, tokenizer

class QuestionCondPairsDataseq(Sequence):
    def __init__(self, dataset, tokenizer, is_train=True, max_len=120, 
                 sampler=None, shuffle=False, batch_size=32):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_len = max_len
        self.sampler = sampler
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()       
    
    def _pad_sequences(self, seqs, max_len=None):
        return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')
    
    def __getitem__(self, batch_id):
        batch_data_indices = \
            self.global_indices[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_data_indices]

        X1, X2 = [], []
        Y = []
        
        for data in batch_data:
            x1, x2 = self.tokenizer.encode(first=data.question.lower(), 
                                           second=data.cond_text.lower())
            X1.append(x1)
            X2.append(x2)
            if self.is_train:
                Y.append([data.label])
    
        X1 = self._pad_sequences(X1, max_len=self.max_len)
        X2 = self._pad_sequences(X2, max_len=self.max_len)
        inputs = {'input_x1': X1, 'input_x2': X2}
        if self.is_train:
            Y = self._pad_sequences(Y, max_len=1)
            outputs = {'output_similarity': Y}
            return inputs, outputs
        else:
            return inputs
                    
    def on_epoch_end(self):
        self.data = self.sampler.sample(self.dataset)
        self.global_indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.global_indices)
    
    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)    

def merge_result(qc_pairs, result, task1_result_dict):
    select_result = defaultdict(set)
    pre_query_id = ""
    cond_sql_score_dict = {}
    last_qry_id = ""
    for pair, score in zip(qc_pairs, result):
        if pre_query_id == "":
            pre_query_id = pair.query_id
        if pre_query_id != pair.query_id:            
            sorted_dict = dict(sorted(cond_sql_score_dict.items(), key=lambda x: x[1], reverse=True))
            if task1_result_dict[pre_query_id]['cond_conn_op'] == 0:
                cond_sql = list(sorted_dict.keys())[0]
                select_result[pre_query_id].update([cond_sql])
            else:
                try:
                    if task1_result_dict[pre_query_id]['cond_conn_op'] == 1:
                        loop_range = 2
                    else:
                        loop_range = task1_result_dict[pre_query_id]['cond_conn_op']
                    for i in range(0, loop_range):
                        cond_sql = list(sorted_dict.keys())[i]
                        select_result[pre_query_id].update([cond_sql])
                except Exception as e:
                    print(pre_query_id, task1_result_dict[pre_query_id]['cond_conn_op'],e)
            pre_query_id = pair.query_id
            cond_sql_score_dict = {}
        cond_sql_score_dict[pair.cond_sql] = score
        last_qry_id = pair.query_id
    
    sorted_dict = dict(sorted(cond_sql_score_dict.items(), key=lambda x: x[1], reverse=True))
    if task1_result_dict[last_qry_id]['cond_conn_op'] == 0:
        cond_sql = list(sorted_dict.keys())[0]
        select_result[last_qry_id].update([cond_sql])
    else:
        try:
            if task1_result_dict[last_qry_id]['cond_conn_op'] == 1:
                loop_range = 2
            else:
                loop_range = task1_result_dict[last_qry_id]['cond_conn_op']
            for i in range(0, loop_range):
                cond_sql = list(sorted_dict.keys())[i]
                select_result[last_qry_id].update([cond_sql])
        except Exception as e:
            print(last_qry_id, task1_result_dict[last_qry_id]['cond_conn_op'],e)
    
    return dict(select_result)

def get_task1_model():
    # Download pretrained BERT model from https://github.com/ymcui/Chinese-BERT-wwm
    bert_model_path = '../model'
    paths = get_checkpoint_paths(bert_model_path)

    num_sel_agg = len(SQL.agg_sql_dict) + 1
    num_cond_op = len(SQL.op_sql_dict) + 1
    num_cond_conn_op = len(SQL.conn_sql_dict)

    bert_model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=None)

    inp_token_ids = Input(shape=(None,), name='input_token_ids', dtype='int32')
    inp_segment_ids = Input(shape=(None,), name='input_segment_ids', dtype='int32')
    inp_header_ids = Input(shape=(None,), name='input_header_ids', dtype='int32')
    inp_header_mask = Input(shape=(None, ), name='input_header_mask')

    x = bert_model([inp_token_ids, inp_segment_ids]) # (None, seq_len, 768)

    # predict cond_conn_op
    x_for_cond_conn_op = Lambda(lambda x: x[:, 0])(x) # (None, 768)
    p_cond_conn_op = Dense(num_cond_conn_op, activation='softmax', name='output_cond_conn_op')(x_for_cond_conn_op)

    # predict sel_agg
    x_for_header = Lambda(seq_gather, name='header_seq_gather')([x, inp_header_ids]) # (None, h_len, 768)
    header_mask = Lambda(lambda x: K.expand_dims(x, axis=-1))(inp_header_mask) # (None, h_len, 1)

    x_for_header = Multiply()([x_for_header, header_mask])
    x_for_header = Masking()(x_for_header)

    p_sel_agg = Dense(num_sel_agg, activation='softmax', name='output_sel_agg')(x_for_header)

    x_for_cond_op = Concatenate(axis=-1)([x_for_header, p_sel_agg])
    p_cond_op = Dense(num_cond_op, activation='softmax', name='output_cond_op')(x_for_cond_op)

    model = Model(
        [inp_token_ids, inp_segment_ids, inp_header_ids, inp_header_mask],
        [p_cond_conn_op, p_sel_agg, p_cond_op]
    )

    model_path = ('task1_0510_best_model_V2.h5')
    model.load_weights(model_path)
    return model

def get_task2_model():
    bert_model_path = '../model'
    paths = get_checkpoint_paths(bert_model_path)
    model, tokenizer = construct_model(paths)
    model_path = 'task2_0510_best_model_V2.h5'
    model.load_weights(model_path)
    return model, tokenizer

def load_test_data():
    abs_path = '../TableQA/'

    test_table_file = abs_path + 'customer/customer.tables.json'
    test_data_file = abs_path + 'customer/data.json'

    test_tables = read_tables(test_table_file)
    test_data = read_data(test_data_file, test_tables)
    return test_tables, test_data

def load_val_data():
    abs_path = '../TableQA/'
    val_table_file = abs_path + 'val/val.tables.json'
    val_data_file = abs_path + 'val/val.json'

    val_tables = read_tables(val_table_file)
    val_data = read_data(val_data_file, val_tables)
    return val_data

def get_val_seq():
    bert_model_path = '../model'
    paths = get_checkpoint_paths(bert_model_path)
    token_dict = load_vocabulary(paths.vocab)
    query_tokenizer = QueryTokenizer(token_dict)
    label_encoder = SqlLabelEncoder()
    val_data = load_val_data()
    val_dataseq = DataSequence(
        data=val_data, 
        tokenizer=query_tokenizer,
        label_encoder=label_encoder,
        is_train=False, 
        shuffle_header=False,
        max_len=160, 
        shuffle=False,
        batch_size=16
    )
    return val_dataseq

def main_process(task1_model, task2_model, tokenizer, val_dataseq):
    bert_model_path = '../model'
    paths = get_checkpoint_paths(bert_model_path)
    token_dict = load_vocabulary(paths.vocab)    
    query_tokenizer = QueryTokenizer(token_dict)
    label_encoder = SqlLabelEncoder()
    
    test_tables, test_data = load_test_data()

    test_dataseq = DataSequence(
        data=test_data, 
        tokenizer=query_tokenizer,
        label_encoder=label_encoder,
        is_train=False, 
        shuffle_header=False,
        max_len=160, 
        shuffle=False,
        batch_size=16
    )

    pred_sqls = []

    for batch_data in tqdm(test_dataseq):
        header_lens = np.sum(batch_data['input_header_mask'], axis=-1)
        preds_cond_conn_op, preds_sel_agg, preds_cond_op = task1_model.predict_on_batch(batch_data)
        sqls = outputs_to_sqls(preds_cond_conn_op, preds_sel_agg, preds_cond_op, header_lens, val_dataseq.label_encoder)
        pred_sqls += sqls

    task1_output_file = 'task1_customer_output.json'
    with open(task1_output_file, 'w') as f:
        for sql in pred_sqls:
            json_str = json.dumps(sql, ensure_ascii=False)
            f.write(json_str + '\n')

    task1_file = 'task1_customer_output.json'

    task1_result = load_json(task1_file)

    te_qc_pairs = QuestionCondPairsDataset(test_data, 
                                        candidate_extractor=CandidateCondsExtractor(share_candidates=True),
                                        has_label=False,
                                        model_1_outputs=task1_result)

    te_qc_pairs_seq = QuestionCondPairsDataseq(te_qc_pairs, tokenizer, 
                                            sampler=FullSampler(), shuffle=False, batch_size=128)

    te_result = task2_model.predict_generator(te_qc_pairs_seq, verbose=1)

    task1_result_dict = {}
    for query_id, pred_sql in enumerate(task1_result):
        task1_result_dict[query_id] = pred_sql

    task2_result = merge_result(te_qc_pairs, te_result, task1_result_dict)  

    for query_id, pred_sql in enumerate(task1_result):
        cond = list(task2_result.get(query_id, []))
        pred_sql['conds'] = cond

    print(pred_sql)

    op_sql_dict = {0: ">", 1: "<", 2: "=", 3: "!="}
    agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
    conn_sql_dict = {0: "", 1: "and", 2: "or"}

    cols = test_tables['6c11e43d2fb211ebbb1a7315e321f5c5'].header

    table_schema = {0:"main_category", 1:"sub_category", 2:"product_type", 3:"weight", 4:"item", 5:"price", 6:"unit_price", 7:"location"}
    ol_mapping_table = {"main_category":"主分類", "sub_category":"次分類", "product_type":"商品種類", "weight":"重量", "item":"品項", "price":"價格", "unit_price":"單價", "location":"位置"}
    
    result_sqls = []
    for query_id, pred_sql in enumerate(task1_result):
        col = ''
        cond = list(task2_result.get(query_id, []))
        print(cond)
        pred_sql['conds'] = cond
        sel_str = 'select '
        
        for agg in pred_sql['agg']:
            if agg > 0:
                sel_str = sel_str + agg_sql_dict[agg] + "("
        
        for sel in pred_sql['sel']:
            if col == '':
                # col = cols[sel][0]
                col = table_schema[sel]
            else:
                col = col + ',' + table_schema[sel]
        
        if "(" in sel_str:
            sel_str = sel_str + col + ') from products where '
        else:
            sel_str = sel_str + col + ' from products where '
        
        cond_cnt = 0
        for conds in pred_sql['conds']:
            col = table_schema[conds[0]]
            op = op_sql_dict[conds[1]]

            if cond_cnt > 0:
                sel_str = sel_str + conn_sql_dict[pred_sql['cond_conn_op']] + " "
            
            if cols[conds[0]][1] != "real":
                sel_str = sel_str + col + " " + op + " '" + conds[2] + "' " 
            else:
                sel_str = sel_str + col + " " + op  + conds[2] + " "
            
            cond_cnt+=1
        
        # idx = sel_str.rfind("and")  # 找到最後一個and的位置
        # if idx != -1:  # 如果找到了and
        #     sel_str = sel_str[:idx] + ";"  # 將字串拆成兩部分，然後在中間插入分號
        print(sel_str)
        if "MIN(" in sel_str:
            pos = sel_str.find("MIN(")
            sel_str = sel_str[:pos] + "product_type, location, " + sel_str[pos:]

        pos = sel_str.find('select location from')
        # 檢查是否需要在 select 後面加上 product_type
        if pos >= 0:
            sel_str = sel_str.replace("select ", "select product_type, ", 1)

        pos = sel_str.find('select item from')
        # 檢查是否需要在 select 後面加上 product_type
        if pos >= 0:
            sel_str = sel_str.replace("select item", "select product_type, location, price ", 1)
            sel_str += " order by price"

        pos = sel_str.find('select price from')
        # 檢查是否需要在 select 後面加上 product_type
        if pos >= 0:
            sel_str = sel_str.replace("select price", "select product_type, location, price ", 1)
            sel_str += " order by price"

        pos = sel_str.find('order')
        if pos < 0:
            sel_str += " order by 1"

        # 使用正規表達式搜尋select和from之間的字串
        sel_cols = re.search('select (.*) from', sel_str)
        
        # 如果有找到搜尋結果，將結果用逗號分割
        if sel_cols:
            selected_cols = sel_cols.group(1).split(', ')  

        for key, value in agg_sql_dict.items():
            for i in range(0, len(selected_cols)):
                selected_cols[i] = selected_cols[i].replace(value, "").replace(")", "").replace("(", "")
            
        result_sqls.append(sel_str)
        selected_cols = [col_mapping_table[col.strip()] for col in selected_cols]
    return result_sqls, selected_cols

if __name__ == '__main__':
    main_process()