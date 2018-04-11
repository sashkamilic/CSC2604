#import gensim
import gc
import itertools
import pandas as pd
import scipy
import tabulate
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import brown, reuters
from lab1 import m1, test, pmi
from sklearn.decomposition import TruncatedSVD


def test2(model, analogy_file):
    '''
    Return the accuracy score on the analogy test
    '''
    acc = 0
    total = 0
    with open(analogy_file) as f:
        for line in f:
            total += 1
            w1,w2,w3,w4 = line.strip().split()
            vector = model[w2] - model[w1] + model[w3]
            result = model.similar_by_vector(vector, topn=1)
            if result[0][0] == w4:
                acc += 1

    return acc / total


def filter_word_test1(M):
    '''
    Create a new word similarity file that only contains words in M
    '''
    filename = '/home/class_test/sasa/RG_word_sims.tsv'
    indices = M.index.values
    out = open('RG_word_sims.filtered.tsv', 'w')
    with open(filename) as f:
        for line in f:
            w1,w2,_ = line.strip().split()
            if all([w in M.index.values for w in [w1,w2]]):
                out.write(line)
    out.close()


def filter_word_test2(M):
    '''
    Create a new word analogy file that only contains words in M
    '''
    filename = '/home/class_test/word2vec_pretrain_vec/word-test.v1.txt'
    indices = M.index.values
    out = open('word-test.v1.filtered.txt', 'w')
    with open(filename) as f:
        for line in f:
            if line[0] in [':', '/']:
                continue
            w1,w2,w3,w4 = line.strip().split()
            if all([w in M.index.values for w in [w1,w2,w3,w4]]):
                out.write(line)
    out.close()


if __name__ == "__main__":

    # Results of Analysis^e are in `results1.txt`
    # and were performed by modifying lab1.py

    # store results in table
    # dict of iterables (keys as columns)
    table = {
        'test': ['similarity', 'analogy']
    }

    ###################
    ## CREATE MODELS ##
    ###################

    PRETRAIN_W2V_FILE = "/home/class_test/word2vec_pretrain_vec/GoogleNews-vectors-negative300.bin"

    #flatten = lambda l: [item for sublist in l for item in sublist]
    #RG_words = flatten([line.split()[0:2] for line in open(SIM_FILE).readlines()])
    #M1 = m1([brown, reuters], 5000, include_words=RG_words, preceeding=True)

    M1 = pd.read_pickle('m1.pkl').to_dense()
    #filter_word_test1(M1)
    #filter_word_test2(M1)

    sim_file = "RG_word_sims.filtered.tsv"
    analogy_file  = "word-test.v1.filtered.txt"

    # save 100dim LSA vectors to w2v file
    '''
    M1_plus = pmi(M1)
    del M1 # free up memory
    gc.collect()
    svd = TruncatedSVD(n_components=100, algorithm="arpack")
    M2_100 = svd.fit_transform(scipy.sparse.csr_matrix(M1_plus))
    M2_100 = pd.DataFrame(data=M2_100, index=M1_plus.index.values)
    out = open('M2_100.txt', 'w')
    out.write('{} {}\n'.format(M2_100.shape[0], M2_100.shape[1]))
    M2_100.to_csv(out, sep=' ', header=False)
    '''

    M2_100 = KeyedVectors.load_word2vec_format('M2_100.txt', binary=False)

    table['M2_100'] = []
    corr, pvalue = test(M2_100, sim_file)
    acc = test2(M2_100, analogy_file)
    table['M2_100'].append('{0:.3f} ({0:.3f})'.format(corr, pvalue))
    table['M2_100'].append('{:.3f}'.format(acc))
    del M2_100
    gc.collect()

    # create word2vec models with dim=50,100,300
    for k in [50, 100, 300]:
        model = Word2Vec(
            sentences=list(brown.sents()) + list(reuters.sents()),
            size=k,
            alpha=0.010,
            workers=4,
            negative=20,
            iter=5
        )
        key = 'w2v_{}dim'.format(str(k))
        table[key] = []
        corr, pvalue = test(model, sim_file)
        acc = test2(model, analogy_file)
        table[key].append('{0:.3f} ({0:.3f})'.format(corr, pvalue))
        table[key].append('{:.3f}'.format(acc))
        del model
        gc.collect()

    # load pretrained model
    w2v_model_pre = KeyedVectors.load_word2vec_format(PRETRAIN_W2V_FILE, binary=True)
    table['w2v_pretrained'] = []
    corr, pvalue = test(w2v_model_pre, sim_file)
    acc = test2(w2v_model_pre, analogy_file)
    table['w2v_pretrained'].append('{0:.3f} ({0:.3f})'.format(corr, pvalue))
    table['w2v_pretrained'].append('{:.3f}'.format(acc))

    print(tabulate.tabulate(table, headers='keys', tablefmt="fancy_grid"))


