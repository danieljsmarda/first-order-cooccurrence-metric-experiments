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

########## p-values ###########

def produce_test_statistic(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    '''Calculates test statistic s(X,Y,A,B).'''
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(wv_obj, X_terms, Y_terms, A_terms, B_terms)
    x_associations = np.apply_along_axis(lambda x_vec: calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx), 1, X_mtx)
    y_associations = np.apply_along_axis(lambda y_vec: calculate_association_metric_for_target_word(y_vec, A_mtx, B_mtx), 1, Y_mtx)
    return np.sum(x_associations) - np.sum(y_associations)

def get_complements(x_union_y):
    '''Generator function that yields pairs of equal-size disjoint subsets
    of x_union_y.
    x_union_y should a set type.'''
    already_seen = set()
    for seq in combinations(x_union_y, len(x_union_y)//2):
        complement = frozenset(x_union_y.difference(seq))
        #already_seen.append(complement)
        already_seen.add(complement)
        #if frozenset(seq) in already_seen:
            #continue
        yield (seq, complement)

def produce_p_value(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    '''Generates the p-value for a set of terms with the word-vector object.
    High-level function; this function should be directly imported into 
    notebooks for experimentation.'''
    x_union_y = set(X_terms).union(set(Y_terms))
    total_terms = len(x_union_y)
    total_pairs = 0
    high_test_statistics = 0
    comparison_statistic = produce_test_statistic(wv_obj, X_terms, Y_terms, A_terms, B_terms)
    for (X_i_terms, Y_i_terms) in tqdm(get_complements(x_union_y), total=num_combinations(total_terms, total_terms/2)):
        total_pairs += 1
        test_statistic = produce_test_statistic(wv_obj, X_i_terms, Y_i_terms, A_terms, B_terms)
        if (test_statistic > comparison_statistic): high_test_statistics += 1
    return (float(high_test_statistics) / float(total_pairs))