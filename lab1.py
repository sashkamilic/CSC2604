import nltk
import re
from collections import Counter, defaultdict
from nltk.corpus import brown
from nltk.util import ngrams
from multiprocessing.dummy import Pool as ThreadPool
#import pandas as pd
from pandas import *


def m1(corpus, n):
    '''
    Construct a word-context vector model by collecting
    bigram counts for top-n words in corpus.
    '''
    # Extract the 5000 most common English words
    freq_dict = nltk.FreqDist(w.lower() for w in corpus.words() if re.match('\w', w))
    W = list(dict(freq_dict.most_common(n)).keys())

    # Get bigrams from words in W
    bigrams = ngrams([w.lower() for w in corpus.words()], 2)
    bigrams = [t for t in bigrams if t[0] in W and t[1] in W]

    # convert bigram counts to cooccurance matrix
    c = Counter(bigrams)
    d = defaultdict(lambda : defaultdict(int))
    for t,freq in c.items():
        d[t[1]][t[0]] = freq

    M1 = DataFrame(d).fillna(0)
    return M1


if __name__ == "__main__":

    #M1 = m1(brown, 5000)
    #M1.to_pickle('m1.pkl')
    M1 = pandas.read_pickle('m1.pkl')
    # convert matrix into probabilities
    P_x = M1.sum(axis=1) / M1.values.sum()
    print(M1.values.sum())
    print(P_x)
