import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import TFBertModel, BertConfig
from tensorflow.keras.optimizers import Adam

import math

def get_swot_data():
    with open('../data/swot_analysis_data.json') as f:
        data = json.load(f)
    return data

def get_lables():
    lable_mapping = {
        "weaknesses": 0,
        "opportunities" :1,
        "strengths": 2,
        "threats": 3
    }
    return lable_mapping

def get_texts_labels(data):
    lable_mapping = get_lables()
    texts, labels = [], []
    for e in data:
        if e["word_count"] < 200:
            texts.append(e['description'])
            labels.append(lable_mapping[e['type']])
    return texts, labels


def split_data(texts,labels):
    texts_df = pd.DataFrame.from_dict({
        'texts' : texts,
        'labels' : labels
        })
    n_classes = texts_df['labels'].nunique()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=.1,
        random_state=42,
        shuffle=True)

    # train_pd = pd.DataFrame.from_dict({
    #     'texts' : train_texts,
    #     'labels' : train_labels
    #     })

    # print(train_pd['labels'].value_counts())
    return train_texts, test_texts, train_labels, test_labels, n_classes


def get_max_len_filter_data(texts, labels, tokenizer):
    _texts, _labels = [], []
    max_len = 0
    bert_max_len = 350
    for id, sent in enumerate(texts):
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
        if len(input_ids) < bert_max_len:
            _texts.append(sent)
            _labels.append(labels[id])
    print('max_len :', max_len)
    print('train_len :', len(_texts))
    print('considered max_len :', bert_max_len)
    return _texts, _labels, bert_max_len


def tokenize_dataset(texts, labels, tokenizer, max_len):
    input_ids, attention_mask_ids, token_type_ids = [], [], []
    for text in texts:
        encoded_text = tokenizer.encode_plus(text,
                            add_special_tokens = True,
                            max_length = max_len,
                            padding='max_length',
                            return_attention_mask = True,
                            return_token_type_ids=True,
                            truncation = True)

        input_ids.append(encoded_text['input_ids'])
        attention_mask_ids.append(encoded_text['attention_mask'])
        token_type_ids.append(encoded_text['token_type_ids'])

    return np.array(input_ids), np.array(attention_mask_ids), np.array(token_type_ids)



def get_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer


def get_model(n_classes, max_len):
    config = BertConfig(num_labels= n_classes)
    transformer_model = TFBertModel.from_pretrained('bert-base-uncased', config = config)

    input_ids_in = tf.keras.layers.Input(shape=(max_len,), name="input_ids", dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(max_len,), name="attention_mask_ids", dtype='int32')
    input_token_in = tf.keras.layers.Input(shape=(max_len,), name="token_type_ids", dtype='int32')

    embedding_layer = transformer_model(input_ids_in, attention_mask=input_masks_in, token_type_ids= input_token_in,)[0]
    x = tf.keras.layers.Dense(256, activation='relu')(embedding_layer[:, 0, :])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    X = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=[input_ids_in, input_masks_in, input_token_in], outputs = X)

    optimizer = Adam(lr = 8e-5)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer= optimizer,
        metrics=["sparse_categorical_accuracy"])

    return model



def fit_model(model, train_dataset, train_labels):
    history = model.fit(
        {
            "input_ids": train_dataset[0],
            "attention_mask_ids": train_dataset[1],
            "token_type_ids": train_dataset[2]
        },
        np.array(train_labels),
        epochs = 3,
        batch_size = 16)

    return model

def predict_model(model, test_dataset, test_labels):
    print("Predict Model")
    y_pred = [np.argmax(i) for i in model.predict(test_dataset)]
    con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))
    con_mat_df = pd.DataFrame(con_mat_norm,index=label_names,columns=label_names)
    print(con_mat_df)

def main(train_model = False):
    swot_data = get_swot_data()
    tokenizer = get_tokenizer()
    texts, labels = get_texts_labels(swot_data)
    # texts, labels, max_len = get_max_len_filter_data(texts, labels, tokenizer)
    max_len = 350
    train_texts, test_texts, train_labels, test_labels, n_classes = split_data(texts,labels)
    # train_dataset = tokenize_dataset(train_texts, train_labels, tokenizer, max_len)
    test_dataset = tokenize_dataset(test_texts, test_labels, tokenizer, max_len)

    model = get_model(n_classes,max_len)
    model.load_weights('swot_analysis_bert_weights').expect_partial()

    predict_model(model, test_dataset, test_labels)
    print("Deletng Model")
    del model
    print ("DONE")



if '__main__' == __name__:
    main()
