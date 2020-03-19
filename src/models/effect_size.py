from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine as cosine_distance
import numpy as np

def word_set_to_mtx(wv_obj, word_set):
    '''Converts set of string words into a 2-D numpy array of word vectors from the word-vector object.'''
    return np.vstack(tuple(wv_obj[word] for word in word_set))

def get_matrices_from_term_lists(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    '''Uses wv_obj to convert lists of words to arrays of word vectors.
    Returns: list of matrices containing the corresponding vectors for 
    the words in X_terms, Y_terms, A_terms, and B_terms.'''
    return [word_set_to_mtx(wv_obj, terms) for terms in [X_terms, Y_terms, A_terms, B_terms]]

def calculate_association_metric_for_target_word(word_vec, A_mtx, B_mtx):
    '''Computes the association metric, s(w,A,B).
    word_vec: 1-D word vector
    A_mtx, B_mtx: 2-D word vector arrays'''
    A_cosines = np.apply_along_axis(lambda row: 1-cosine_distance(row, word_vec), 1, A_mtx)
    B_cosines = np.apply_along_axis(lambda row: 1-cosine_distance(row, word_vec), 1, B_mtx)
    return np.mean(A_cosines) - np.mean(B_cosines)

def calculate_effect_size(X_mtx, Y_mtx, A_mtx, B_mtx):
    '''Computes the effect size.
    X_mtx, Y_mtx, A_mtx, B_mtx: 2-D word vector arrays.'''
    x_associations = np.apply_along_axis(lambda x_vec: calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx), 1, X_mtx)
    y_associations = np.apply_along_axis(lambda y_vec: calculate_association_metric_for_target_word(y_vec, A_mtx, B_mtx), 1, Y_mtx)
    X_union_Y = np.vstack((X_mtx, Y_mtx))
    all_associations = np.apply_along_axis(lambda w_vec: calculate_association_metric_for_target_word(w_vec, A_mtx, B_mtx), 1, X_union_Y)
    return (np.mean(x_associations) - np.mean(y_associations))/np.std(all_associations, ddof=1)

def produce_effect_size(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    '''Highest-level function, from word-vector object to output effect size.'''
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(wv_obj, X_terms, Y_terms, A_terms, B_terms)
    return calculate_effect_size(X_mtx, Y_mtx, A_mtx, B_mtx)