import nltk
import re
import scipy
from collections import Counter, defaultdict
from nltk.corpus import brown, reuters, stopwords
from nltk.util import ngrams
from multiprocessing import Pool
#import pandas as pd
from pandas import *
from numpy import log
import time
from functools import partial

#np.set_printoptions(threshold=np.nan)

SIM_FILE = "RG_word_sims.tsv"
NUM_PROCESSES = 8

def m1(corpus, n, include_words=None):
    '''
    Construct a word-context vector model by collecting
    bigram counts for top-n words in corpus.

    `include_words` - optional list of words to include along
    with top-n
    '''
    # Extract the 5000 most common English words
    freq_dict = nltk.FreqDist(w.lower() for w in corpus.words())
    # remove non numeric from freq_dict
    keys = list(freq_dict.keys())
    for k in keys:
        if not re.match('[a-zA-Z]', k):
            del freq_dict[k]

    W = list(dict(freq_dict.most_common(n)).keys())
    W.extend(include_words)

    # Get bigrams from words in W
    bigrams = ngrams([w.lower() for w in corpus.words()], 2)
    # filter out words not in W
    bigrams = [bi for bi in bigrams if bi[0] in W and bi[1] in W]

    # convert bigram counts to cooccurance matrix
    c = Counter(bigrams)
    d = defaultdict(lambda : defaultdict(int))
    for t,freq in c.items():
        d[t[1]][t[0]] = freq

    M1 = DataFrame(d).fillna(0).to_sparse()

    # remove words not in both rows and columns
    M1 = M1.drop(list(set(M1.index) - set(M1.columns.values)))
    M1 = M1.drop(list(set(M1.columns.values) - set(M1.index)), axis=1)

    return M1


def pmi(M):
    # +1 for smoothing
    P_xy = (M + 1) / M.values.sum()
    P_x = (M.sum(axis=1) + 1) / M.values.sum()
    P_y = (M.sum(axis=0) + 1) / M.values.sum()
    return np.log(P_xy.div(P_x, axis=1).div(P_y, axis=0))


def test(M):
    '''
    Compute Pearson correlation between similarity scores in
    Rubenstein and Goodenougth (1965) and cosine similarities in M
    '''
    filelines = [line.strip().split() for line in open(SIM_FILE).readlines()]
    sims_dict = dict([((w1,w2), float(s)) for (w1,w2,s) in filelines])

    x = []
    y = []

    for (w1, w2) in sims_dict.keys():

        if w1 not in M:
            #print('"{}" not in matrix'.format(w1))
            continue

        if w2 not in M:
            #print('"{}" not in matrix'.format(w2))
            continue

        v1 = np.array(M[w1])
        v2 = np.array(M[w2])
        x.append(scipy.spatial.distance.cosine(v1, v2))
        y.append(sims_dict[(w1, w2)])

    x = np.array(x)
    y = np.array(y)

    return scipy.stats.pearsonr(x, y)


def get_col_name(row):
    b = (df.ix[row.name] == row['value'])
    return b.index[b.argmax()]

def filter_nonW(W, l):
    list(filter(lambda t: t[0] in W and t[1] in W, l))

if __name__ == "__main__":

    freq_dict = nltk.FreqDist(w.lower() for w in brown.words())
    words = set([line.split()[0] for line in open(SIM_FILE).readlines()])
    # include words (freq > 10) from R&G sim list along with top-5000 in M1
    additional_words = [k for k in words if freq_dict[k] >= 5]
    print(additional_words)
    M1 = m1(brown, 5000, include_words=additional_words)
    #M1.to_pickle('m1.pkl')
    M1 = pandas.read_pickle('m1.pkl').to_dense()
    M1_plus = pmi(M1)
    # try averaging cosign of context and word vectors (i.e. row/column vectors)
    print(test(M1))
    print(test(M1_plus))
