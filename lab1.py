import itertools
import nltk
import numpy as np
import pandas as pd
import re
import scipy
from collections import Counter, defaultdict
from multiprocessing import Pool
from nltk.corpus import brown, reuters
from nltk.util import ngrams
from tabulate import tabulate
#import pandas as pd
from sklearn.decomposition import TruncatedSVD

SIM_FILE = "RG_word_sims.tsv"


def m1(corpora, n, include_words=None):
    '''
    Construct a word-context vector model by collecting
    bigram counts for top-n words in the corpora.

    `include_words` - optional list of words to include along
    with top-n
    '''
    corpora_iter = itertools.chain.from_iterable([c.words() for c in corpora])
    # Extract the 5000 most common English words
    freq_dict = nltk.FreqDist(w.lower() for w in corpora_iter)
    # remove non numeric from freq_dict
    keys = list(freq_dict.keys())
    for k in keys:
        if not re.match('[a-zA-Z]', k):
            del freq_dict[k]

    W = list(dict(freq_dict.most_common(n)).keys())
    if include_words:
        W.extend(include_words)

    # Get bigrams from words in W
    corpora_iter = itertools.chain.from_iterable([c.words() for c in corpora])
    bigrams = ngrams([w.lower() for w in corpora_iter], 2)
    # filter out words not in W
    bigrams = [bi for bi in bigrams if bi[0] in W and bi[1] in W]

    # convert bigram counts to cooccurance matrix
    c = Counter(bigrams)
    d = defaultdict(lambda : defaultdict(int))
    for t,freq in c.items():
        d[t[1]][t[0]] = freq

    M1 = pd.DataFrame(d).fillna(0).to_sparse()

    # remove words not in both rows and columns
    M1 = M1.drop(list(set(M1.index) - set(M1.columns.values)))
    M1 = M1.drop(list(set(M1.columns.values) - set(M1.index)), axis=1)

    return M1


def pmi(M, k=1, normalized=False):
    '''
    If k > 1, compute PMI^k
    (see: "Handling the Impact of Low frequency Events..." (2011))
    '''
    # +1 for smoothing
    P_xy = (M + 1) / M.values.sum()
    P_xy = P_xy.pow(k)
    P_x = (M.sum(axis=1) + 1) / M.values.sum()
    P_y = (M.sum(axis=0) + 1) / M.values.sum()
    pmi = np.log(P_xy.div(P_x, axis=1).div(P_y, axis=0))
    if normalized:
        return pmi.div(-np.log(P_xy))
    return pmi


def test(M, sims):
    '''
    Compute Pearson correlation between similarity scores in
    `P` (dict in the form (word1, word2) -> float) and and
    cosine similarities in M
    '''
    indices = M.index.values

    x = []
    y = []

    for (w1, w2) in P.keys():

        if w1 not in indices:
            #print('"{}" not in matrix'.format(w1))
            continue

        if w2 not in indices:
            #print('"{}" not in matrix'.format(w2))
            continue

        v1 = np.array(M.loc[[w1]])
        v2 = np.array(M.loc[[w2]])

        x.append(scipy.spatial.distance.cosine(v1, v2))
        y.append(P[(w1, w2)])

    x = np.array(x)
    y = np.array(y)

    return scipy.stats.pearsonr(x, y)


if __name__ == "__main__":

    #corpora_iter = itertools.chain.from_iterable([c.words() for c in [brown, reuters]])
    #freq_dict = nltk.FreqDist(w.lower() for w in corpora_iter)
    #words = [line.split()[0] for line in open(SIM_FILE).readlines()]
    #include_words = [w for w in words if freq_dict[w] >= 15]

    #M1 = m1([brown, reuters], 5000)
    #M1.to_pickle('m1.pkl')

    filelines = [line.strip().split() for line in open(SIM_FILE).readlines()]
    P = dict([((w1,w2), float(s)) for (w1,w2,s) in filelines])

    # co-occurrance matrix
    M1 = pd.read_pickle('m1.pkl').to_dense()
    # pmi
    M1_plus = pmi(M1)
    # pmi^2
    M1_pmi2 = pmi(M1, 2)
    # npmi
    M1_npmi = pmi(M1, normalized=True)
    # ppmi
    M1_ppmi = M1_plus.copy()
    M1_ppmi[M1_ppmi < 0] = 0

    names = ['raw', 'pmi', 'pmi^2', 'npmi', 'ppmi']
    Ms = [M1, M1_plus, M1_pmi2, M1_npmi, M1_ppmi]

    header = ['measure', 'M1', 'M2_10', 'M2_20', 'M2_50', 'M2_100']
    table = []
    for name, M in zip(names, Ms):

        result = [name]
        # first test on non-truncated matrix
        corr, pvalue = test(M, P)
        result.append('{0:.3f} ({0:.3f})'.format(corr, pvalue))

        for k in [10, 20, 50, 100]:
            svd = TruncatedSVD(n_components=k, algorithm="arpack")
            M_trunc = svd.fit_transform(scipy.sparse.csr_matrix(M))
            M_trunc = pd.DataFrame(data=M_trunc, index=M1.index.values)
            corr, pvalue = test(M_trunc)
            result.append('{0:.3f}'.format(corr))

        table.append(result)

    print(tabulate(table, header, tablefmt="fancy_grid"))

