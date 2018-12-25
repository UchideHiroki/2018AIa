from copy import copy, deepcopy
from random import sample
from collections import defaultdict
from pathlib import Path
import networkx as nx
import pickle
from itertools import permutations
import numpy as np
from random import sample
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
import gc
import time
import warnings
from contextlib import contextmanager
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {}'.format(title, time.time() - t0))


class Morph:


    def __init__(self, embedding):
        self.embedding = embedding
        self.rules = None
        self.transformations = None
        self.transformations_evaluated = None
        self.graph = None
        self.graph_normalized = None

    def extract_rule(self, rules, word1, word2, max_len=6):
        # check suffix
        i = 1
        while (word1[:i] == word2[:i]):
            i += 1
        if i != 1and i > max(len(word1[i-1:]), len(word2[i-1:])) <= max_len:
            rules[('suffix', word1[i-1:], word2[i-1:])].append((word1, word2))

        # check prefix
        i = 1
        while (word1[-i:] == word2[-i:]):
            i += 1
        if i != 1 and i > max(len(word1[:-i+1]), len(word2[:-i+1])) <= max_len:
            rules[('prefix', word1[:-i+1], word2[:-i+1])].append((word1, word2))

    def extract_rules(self, vocab, read=True, save=False, file_name=None):
        '''
        Extract candidate prefix/suffix rules from V.

        input
        vocab(set): set of words
        real(bool):
        save(bool):
        file_name: hogehoge.pickle    '../preprocessed/FILE_NAME'

        output
        self.rules(dict):
            key(tuple): ('prefix/suffix', 'ing', 'ed')
            value(list): list of word_pair (word1, word2) supported by rule
        '''
        if file_name:
            path = Path.cwd().parent.joinpath('preprocessed/{}'.format(file_name))
        else:
            path = Path.cwd().parent.joinpath('preprocessed/rules.pickle')
        if path.exists() and read:
            with path.open('rb') as f:
                self.rules = pickle.load(f)
        else:
            self.rules = defaultdict(list)
            for (word1, word2) in tqdm(permutations(vocab, 2)):
                self.extract_rule(self.rules, word1, word2)
        if save:
            with path.open('wb') as f:
                pickle.dump(self.rules, f)

    def downsample_rules(self, n_sample=1000):
        '''
        downsample the number of word pair in suppor set of each rule.
        '''
        for key, value in self.rules.items():
            if len(value) >= n_sample:
                self.rules[key] = sample(value, n_sample)

    def is_word_pairs_similar(self, word_pair1, word_pair2, annoy_index=None, topn=10):
        '''
        Calculate if
            word2_1 + (word1_2 - word1_1)
        is similar to word2_2

        input
            word_pair1(tuple): (word1_1, word1_2), used as direction vector
            word_pair2(tuple): (word2_1, word2_2)

        output
            bool
        '''
        closest_n = self.embedding.most_similar(
            positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]],
                                                                         topn=topn, indexer=annoy_index)
        for word, cos_sim in closest_n:
            if word == word_pair2[1]:
                return True
        return False

    def get_similarity_rank(self, word_pair1, word_pair2, topn=100):
        '''
        Calculate similarity rank and cosine similarity

        input
            word_pair1(tuple): direction vector
            word_pair2(tuple): word pair to calculate similarity
        return
            tuple (rank, cos_sim)
        '''
        closest_n = self.embedding.most_similar(
            positive=[word_pair2[0], word_pair1[1]], negative=[word_pair1[0]], topn=topn)
        for rank, (word, cos_sim) in enumerate(closest_n):
            if word == word_pair2[1]:
                return (rank+1, cos_sim) # add 1 to the highest rank to 1, not 0
        return (None, None)

    def index_vector(self, dimensions=300, save=False):
        '''
        make annoy_index which is used in function 'is_word_pairs_similar'
        Using annoy_index, execution may be slower than normal index
        '''
        path = Path.cwd().parent.joinpath('preprocessed/annoy.index')
        if path.exists():
            annoy_index = AnnoyIndexer()
            annoy_index.load(str(path))
            annoy_index.model = self.embedding
        else:
            annoy_index = AnnoyIndexer(self.embedding, dimensions)
            if save:
                annoy_index.save(str(path))
        return annoy_index

    def get_best_transformation(self, rule, support_set, topn=100, return_hit_rate=False):
        '''
        Get the most supportable transformation from the rule.

        input
            rule(tuple): (type, from, to)
            support_set(set): {(word1, word2), ...}
            topn(int): compute rank and cos_sim if rank is under topn

        return
            transformation(tuple): ('suffix', 'ing', 'ed', w, w')
            trans_support_set(set): {((word1, word2), rank, cosine), ...}
            hit_rate(real): optional
        '''
        # Calculate all the transformations
        transformations = defaultdict(tuple)
        for word_pair1 in support_set:
            transformation = rule + word_pair1  # tuple(type, from, to, w, w')
            trans_support_set = set()
            n_hit = 0
            for word_pair2 in support_set:
                if word_pair2 != word_pair1:
                    (rank, cos_sim) = self.get_similarity_rank(word_pair1, word_pair2, topn=topn)
                    trans_support_word = (word_pair2, rank, cos_sim)
                    if rank is not None:
                        n_hit += 1
                        trans_support_set.add(trans_support_word)
            # avoid 0 division
            hit_rate = n_hit / \
                len(trans_support_set) if len(trans_support_set) != 0 else 0
            transformations[transformation] = (hit_rate, trans_support_set)

        # Select hit_rate-highest one
        transformations_by_count = sorted(
            transformations.items(), key=lambda kv: kv[1][0], reverse=True)
        best_transformation, (best_hit_rate,
                              trans_support_set) = transformations_by_count[0]

        if return_hit_rate:
            return best_transformation, trans_support_set, best_hit_rate
        else:
            return best_transformation, trans_support_set

    def generate_transformations(self, topn=100, min_explain_count=10):
        '''
        Generate transformations and their support_sets

        return
            transformations(dict):
                key: transformation (type, from, to, word1, word2)
                value: transformation-support set {((word1, word2), rank, cosine), ...}
        '''
        transformations = defaultdict(dict)
        for rule, support_set in tqdm(self.rules.items()):
            support_set = set(support_set)
            while True:
                (tran, tran_support_set) = self.get_best_transformation(rule, support_set, topn=topn)
                if len(tran_support_set) < min_explain_count:
                    break
                transformations[tran] = tran_support_set
                tran_support_words = {word[0] for word in tran_support_set}
                support_set = support_set - {tran[3:5]} - tran_support_words
                if len(support_set) < min_explain_count:
                    break
        self.transformations = transformations

    def generate_trans(self, rule, support_set, topn, min_explain_count):
        trans_on_rule = defaultdict(dict)
        support_set = set(support_set)
        while True:
            (tran, tran_support_set) = self.get_best_transformation(
                rule, support_set, topn=topn)
            if len(tran_support_set) < min_explain_count:
                break
            trans_on_rule[tran] = tran_support_set
            tran_support_words = {word[0] for word in tran_support_set}
            support_set = support_set - {tran[3:5]} - tran_support_words
            if len(support_set) < min_explain_count:
                break
        return trans_on_rule

    def generate_trans_wrapped(self, args):
        return self.generate_trans(*args)

    def generate_transformations_parallel(self, topn=100, min_explain_count=10):
        '''
        Generate transformations and their support_sets in parallel

        return
            transformations(dict):
                key: transformation (type, from, to, word1, word2)
                value: transformation-support set {((word1, word2), rank, cosine), ...}
        '''
        transformations = defaultdict(dict)
        args =[(rule, support_set, topn, min_explain_count) for rule, support_set in self.rules.items()]
        p = Pool() # processesを指定しなければ、cpuの指定出来る最大コア数を使う
        trans_list = p.map(self.generate_trans_wrapped, args)
        [transformations.update(trans) for trans in trans_list]
        self.transformations = transformations

    def evaluate_transformations(self, rank=30, cos_sim=0.5):
        '''
        Evaluate how well it passes a proximity test in embedding space.

        return:
            self.evaluated_transformations
        '''
        trans = deepcopy(self.transformations)
        for tran, support_set in tqdm(self.transformations.items()):
            for word in support_set:
                if word[1] > rank or word[2] < cos_sim: # cos_simの制約で結構落ちる
                    trans[tran].remove(word)
        self.transformations_evaluated = trans

    def build_graph(self, is_use_trans_evaluated=True):
        '''
        build Graph and add Nodes and Edges to self.graph

        input:
            is_use_trans_evaluated(bool)
        '''
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.embedding.vocab.keys())
        if is_use_trans_evaluated:
            transformations = self.transformations_evaluated
        else:
            transformations = self.transformations
        for dw, support in tqdm(transformations.items()):
            # word_pair = dw so assign 0 for rank
            G.add_edge(dw[3], dw[4], transformation=dw, rank=0, cos_sim=1)
            for ((word1, word2), rank, cos_sim) in support:
                G.add_edge(word1, word2, transformation=dw, rank=rank, cos_sim=cos_sim)
        self.graph = G

    def normalize_graph(self):
        '''
        convert multi-directed self.graphraph to directed self.graphraph
        '''
        graph = deepcopy(self.graph)
        for node in tqdm(self.graph.nodes()):
            for neighbor in self.graph.neighbors(node):
                # if node and neighbor do not have multiple edges, continue
                if not (self.graph.has_edge(node, neighbor) and self.graph.has_edge(neighbor, node)):
                    continue
                # Delete one of the edges
                # edge w1→w2 considered only if count(w1) < count(w2)
                if self.embedding.vocab[node].count > self.embedding.vocab[neighbor].count:
                    graph.remove_edge(node, neighbor)
                    continue
                # chose the one with minimal rank
                if self.graph[node][neighbor][0]['rank'] > self.graph[neighbor][node][0]['rank']:
                    graph.remove_edge(node, neighbor)
                    continue
                # chose the one with the maximal cosine
                if self.graph[node][neighbor][0]['cos_sim'] < self.graph[neighbor][node][0]['cos_sim']:
                    graph.remove_edge(node, neighbor)
                    continue
            self.graph_normalized = graph

    def get_morph_trans(self, use_normalized_graph=True):
        '''
        Extract all transformations from the graph.

        input:
            use_normalized_graph(bool): use self.graph_normalized if True or use self.graph

        output:
            morph_trans: morphological transformations (type, from, to, w, w')
        '''
        morph_trans = set()
        if use_normalized_graph is True:
            for edge in self.graph_normalized.edges(data=True):
                morph_trans.add(edge[2]['transformation'])
        else:
            for edge in self.graph.edges(data=True):
                morph_trans.add(edge[2]['transformation'])
        return morph_trans


if __name__ == '__main__':
        # limitを5000以上にすると計算量が爆発的に増えて処理が終わらなくなる
        embedding = KeyedVectors.load_word2vec_format('../input/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=100000)
        morph = Morph(embedding)
        with timer('Extract rules...'):
            morph.extract_rules(embedding.vocab.keys(), read=False, save=True, file_name='rules_100000.pickle')
            with open('../preprocessed/rules_100000.pickle', mode='wb') as f:
                pickle.dump(morph.rules, f)
        with timer('Downsample rules...'):
            morph.downsample_rules()
            with open('../preprocessed/downsampled_rules_100000.pickle', mode='wb') as f:
                pickle.dump(morph.rules, f)
        with timer('Generate transformations...'):
            morph.generate_transformations_parallel(topn=10, min_explain_count=4)
            with open('../preprocessed/transformations_100000.pickle', mode='wb') as f:
                pickle.dump(morph.transformations, f)
        with timer('Evaluate transformations...'):
            morph.evaluate_transformations(rank=3, cos_sim=0.5)
        with timer('Building graph...'):
            morph.build_graph(is_use_trans_evaluated=True)
            with open('../preprocessed/graph_100000.pickle', mode='wb') as f:
                pickle.dump(morph.graph, f)
        with timer('Normalize graph...'):
            morph.normalize_graph()
            with open('../preprocessed/normalized_graph_100000.pickle', mode='wb') as f:
                pickle.dump(morph.graph_normalized, f)
