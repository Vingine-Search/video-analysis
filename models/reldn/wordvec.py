# this is used to generate word vector for each word in the vocabulary

import json
import gensim
import numpy as np
from numpy import linalg as la

from config import reader
from typing import List, Dict, Tuple

cfg = reader()

dataset_path = cfg["dataset_dir"]
word_vector_path = cfg["wordvectors_dir"]

# TODO: move to config
objs_path = dataset_path + '/json_dataset/objects.json'
preds_path = dataset_path + '/json_dataset/predicates.json'

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word_vector_path, binary=True)

def lower_all() -> List[str]:
    """
    lower all the words in the word2vec model

    Returns:
        list: all the words in the word2vec model
    """
    vocab = list(word2vec_model.key_to_index.keys())
    for word in vocab:
        word2vec_model.key_to_index[word.lower()] = word2vec_model.key_to_index.pop(word)
    return vocab

def load_objs_cat() -> Dict:
    """
    Returns:
        Dict: of all objects categories in the dataset
    """
    with open(objs_path) as f:
        obj_cats = json.load(f)
    return obj_cats

def load_preds_cat() -> Dict:
    """
    Returns:
        Dict: of all predicates categories in the dataset
    """
    with open(preds_path) as f:
        prd_cats = json.load(f)
    prd_cats.insert(0, 'unknown') # background
    return prd_cats

def get_obj_prd_vecs() -> Tuple[np.ndarray, np.ndarray]:
    lower_all() # preprocess the word2vec model
    obj_cats = load_objs_cat()
    prd_cats = load_preds_cat()

    # change the obj_cat words to vectors and store them in a matrix
    all_obj_vecs = np.zeros((len(obj_cats), 300), dtype=np.float32)
    for r, obj_cat in enumerate(obj_cats):
        obj_words = obj_cat.split()
        for word in obj_words:
            raw_vec = word2vec_model[word]
            all_obj_vecs[r] += (raw_vec / la.norm(raw_vec))
        all_obj_vecs[r] /= len(obj_words)

    # change the prd_cat words to vectors and store them in a matrix
    all_prd_vecs = np.zeros((len(prd_cats), 300), dtype=np.float32)
    for r, prd_cat in enumerate(prd_cats):
        prd_words = prd_cat.split()
        for word in prd_words:
            raw_vec = word2vec_model[word]
            all_prd_vecs[r] += (raw_vec / la.norm(raw_vec))
        all_prd_vecs[r] /= len(prd_words)
    return all_obj_vecs, all_prd_vecs

