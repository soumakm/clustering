# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 18:50:37 2016

@author: Soumak
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import sframe
from sklearn.neighbors import NearestNeighbors

wiki = sframe.SFrame('people_wiki.gl/')
wiki = wiki.add_row_number()             # add row number, starting at 0

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)
    
word_count = load_sparse_csr('people_wiki_word_count.npz')    

map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
#print wiki[wiki['name'] == 'Barack Obama']
distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
#print wiki.join(neighbors, on='id').sort('distance')[['id','name','distance']]

def unpack_dict(matrix, map_index_to_word):
    table = list(map_index_to_word.sort('index')['category'])
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)
def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

obama_words = top_words('Barack Obama')
print obama_words

barrio_words = top_words('Francisco Barrio')
print barrio_words
combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words.sort('Obama', ascending=False)
