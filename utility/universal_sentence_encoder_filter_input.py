import json
import os
import tensorflow_hub as module_hub
import numpy as np


def get_embng_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
    model = module_hub.load(module_url)
    return model

def save_embed(embed_matrix_file_path, embed):
    np.save(embed_matrix_file_path, embed)

def load_embed(embed_matrix_file_path):
    return np.load(embed_matrix_file_path)

def create_embedding_matrix(model,train_data_file_path):
    with open('swot_analysis_data.json') as f:
        data = json.load(f)
    training_data = [x['description'] for x in data][:10000]
    swot_encoding_matrix = embed(model,training_data)
    return swot_encoding_matrix

def create_test_data(train_data_file_path):
    with open('swot_analysis_data.json') as f:
        data = json.load(f)
    test_data = [x['description'] for x in data][40000:]
    return test_data

def train_data_embed_exists(embed_matrix_file_path):
    return bool(os.path.exists(embed_matrix_file_path))

def embed(model, sents):
    return model(sents)

def semantic_similarity(threshold):
    import pdb; pdb.set_trace()
    embed_matrix_file_path =  'train_data_embedding_matrix.npy'
    train_data_file_path = 'swot_analysis_data.json'
    input_sents = create_test_data(train_data_file_path)
    ss_sents = []
    model = get_embng_model()
    if train_data_embed_exists(embed_matrix_file_path):
        swot_embedding_matrix = load_embed(embed_matrix_file_path)
    else:
        swot_embedding_matrix = create_embedding_matrix(model, train_data_file_path)
        save_embed(embed_matrix_file_path, swot_embedding_matrix)
    swot_test_embedding_matrix = embed(model, input_sents)
    corr = np.inner(swot_test_embedding_matrix, swot_embedding_matrix)
    for i, each in enumerate(corr):
        if each.max() >= threshold:
            ss_sents.append(input_sents[i])
    print(len(ss_sents))
    return ss_sents

if __name__ == '__main__':
    ss_sents = semantic_similarity(0.50)

