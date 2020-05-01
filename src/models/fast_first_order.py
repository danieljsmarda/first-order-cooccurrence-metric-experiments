from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine as cosine_distance
from scipy.special import comb as num_combinations
from itertools import combinations
from functools import lru_cache
from tqdm import tqdm
from statistics import mean
import numpy as np
import scipy as sp

def get_complements(x_union_y):
    '''Generator function that yields pairs of equal-size disjoint subsets
    of x_union_y.
    x_union_y should a set type.'''
    for seq in combinations(x_union_y, len(x_union_y)//2):
        complement = frozenset(x_union_y.difference(seq))
        yield (seq, complement)

def get_expSG_1storder_relation_no_cache_NEW(word_from, words_to, we_model):
    ctx_vecs = []
    for _word in words_to:
        _idx = we_model.wv.vocab[_word].index
        ctx_vecs.append(we_model.trainables.syn1neg[_idx])
    ctx_vecs = np.array(ctx_vecs)    
    
    _vec = we_model.wv[word_from]
    relations = sp.special.expit(np.dot(ctx_vecs, _vec))
    
    return relations

def get_expSG_1storder_relation_no_cache_NEW_ALLWORDS(words_to, we_model):
    ctx_vecs = []
    for _word in words_to:
        _idx = we_model.wv.vocab[_word].index
        ctx_vecs.append(we_model.trainables.syn1neg[_idx])
    ctx_vecs = np.array(ctx_vecs)    
    
    _vecs = we_model.wv.vectors
    relations = sp.special.expit(np.dot(_vecs, ctx_vecs.T))
    
    return relations

def get_1storder_association_metric_fast(word, A_terms, B_terms, we_model):
    A_relations = get_expSG_1storder_relation_no_cache_NEW(word, A_terms, we_model)
    B_relations = get_expSG_1storder_relation_no_cache_NEW(word, B_terms, we_model)
    return mean(A_relations) - mean(B_relations)