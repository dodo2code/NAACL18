# -*- coding: utf-8 -*-
#1 vocab_build
import io
import numpy as np
import config
conf = config.config()

general_dir =  '/Users/zy/Desktop/'


cnn_train_dir = general_dir+'query_sum_only_cnn/cnn_data/training/full_vocabulary.txt'
cnn_val_dir   = general_dir+'query_sum_only_cnn/cnn_data/validation/full_vocabulary.txt'
cnn_test_dir  = general_dir+'query_sum_only_cnn/cnn_data/test/full_vocabulary.txt'

daily_train_dir = general_dir+'only_daily/daily_data/training/full_vocabulary.txt'
daily_val_dir   = general_dir+'only_daily/daily_data/validation/full_vocabulary.txt'
daily_test_dir  = general_dir+'only_daily/daily_data/test/full_vocabulary.txt'

def vocab_build(train_dir, val_dir, test_dir):
    vocab_train=[]
    vocab_val=[]
    vocab_test=[]
    
    with io.open(train_dir, encoding='utf-8') as f:
        for line in f:
            word, count = line.split()
            vocab_train.append([word, int(count)])
    
    with io.open(val_dir, encoding='utf-8') as f:
        for line in f:
            word, count = line.split()
            vocab_val.append([word, int(count)])
        
    with io.open(test_dir, encoding='utf-8') as f:
        for line in f:
            word, count = line.split()
            vocab_test.append([word, int(count)])
    
    return vocab_train, vocab_val, vocab_test

cnn_vocab_train, cnn_vocab_val, cnn_vocab_test=vocab_build(cnn_train_dir, cnn_val_dir, cnn_test_dir)
daily_vocab_train, daily_vocab_val, daily_vocab_test=vocab_build(daily_train_dir, daily_val_dir, daily_test_dir)


def merge_dict(dict1, dict2):
    word2id={word_count[0]:index for index, word_count in enumerate(dict1)}
    vocab1 = set([word_count[0] for word_count in dict1])
    add_to_vocab1 = []
    for word_count in dict2:
        if word_count[0] not in vocab1:
            add_to_vocab1.append(word_count)
        else:
            dict1[word2id[word_count[0]]][1]+=word_count[1]
      
    return dict1+add_to_vocab1

cnn_daily_test = merge_dict(cnn_vocab_test, daily_vocab_test)
test_cnn_train = merge_dict(cnn_daily_test, cnn_vocab_train)

full_vocab =  merge_dict(test_cnn_train, daily_vocab_train)
full_vocab=sorted(full_vocab, key=lambda x: x[1], reverse=True)

vocab = [pair[0] for pair in full_vocab]


#2 data_read
special=['<pad>'.encode('utf-8'),'<eos>'.encode('utf-8'),'<s>'.encode('utf-8'), '<unk>'.encode('utf-8')]
source_vocab = special+vocab[:conf.source_vocab_size-len(special)]
target_vocab = source_vocab[:conf.target_vocab_size]

word2id={word:index for index, word in enumerate(source_vocab)}
id2word = {index: word for index, word in enumerate(source_vocab)}
sum_word2id={word:index for index, word in enumerate(target_vocab)}

full_vocabulary = special + vocab
full_word2id ={word:index for index, word in enumerate(full_vocabulary)}
full_id2word = {index: word for index, word in enumerate(full_vocabulary)}


import os
cnn_path_train = general_dir+'query_sum_only_cnn/cnn_data/training/'
cnn_path_val   = general_dir+'query_sum_only_cnn/cnn_data/validation/'
cnn_path_test  = general_dir+'query_sum_only_cnn/cnn_data/test/'

daily_path_train = general_dir+'only_daily/daily_data/training/'
daily_path_val   = general_dir+'only_daily/daily_data/validation/'
daily_path_test  = general_dir+'only_daily/daily_data/test/'

def data_generate(dir_path):
    doc_name = os.listdir(dir_path+'documents/')
    doc=[]
    query=[]
    summ=[]
    for filename in doc_name:
        doc_path = os.path.join(dir_path+'documents/', filename)
        que_path = os.path.join(dir_path+'queries/', filename[:-4]+'.1.txt')
        sum_path = os.path.join(dir_path+'references/', 'A.'+filename[:-4]+'.1.txt')
        
        if os.path.isfile(doc_path):
            with io.open(doc_path, 'r', encoding='utf-8') as f:
                doc.append(f.read().split())
        
        if os.path.isfile(que_path):
            with io.open(que_path, 'r', encoding='utf-8') as f:
                query.append(f.read().split())
        
        if os.path.isfile(sum_path):
            with io.open(sum_path, 'r', encoding='utf-8') as f:
                summ.append(f.read().split())
    
    return doc, query, summ, doc_name


cnn_train_doc, cnn_train_query, cnn_train_summ, cnn_train_doc_name = data_generate(cnn_path_train)
cnn_val_doc,   cnn_val_query,   cnn_val_summ,   cnn_val_doc_name   = data_generate(cnn_path_val)
cnn_test_doc,  cnn_test_query,  cnn_test_summ,  cnn_test_doc_name  = data_generate(cnn_path_test)

daily_train_doc, daily_train_query, daily_train_summ, daily_train_doc_name = data_generate(daily_path_train)
daily_val_doc,   daily_val_query,   daily_val_summ,   daily_val_doc_name   = data_generate(daily_path_val)
daily_test_doc,  daily_test_query,  daily_test_summ,  daily_test_doc_name  = data_generate(daily_path_test)

import rouge
def unk_token(word2id_, word):
    try:
        index=word2id_[word]
    except:
        index=word2id_['<unk>']
    return index
    
def convert_to_id(doc, query, summ, inference=None):
    doc2id=[]   
    query2id=[]
    summ2id=[]
    
    doc_mask=[]
    query_mask=[]
    summ_mask=[]
    
    doc_len=[]
    query_len=[]
    sum_len=[]
    
    sent_seg=[]
    seg_mask=[]
    
    copy_indicator = []
    position = []
    for doc_i, que_i, sum_i in zip(doc, query, summ):
        if len(sum_i)<=conf.sum_max_l:
            if rouge.rouge_n(doc_i[:conf.doc_max_l], sum_i, n=1)[-1]>0.5 and \
               rouge.rouge_n(doc_i[:conf.doc_max_l], sum_i, n=2)[-1]>0.0:
                   
                   doc_len.append(len(doc_i[:conf.doc_max_l]))
                   doc_mask.append([1]*len(doc_i[:conf.doc_max_l])+[0]*(conf.doc_max_l-len(doc_i[:conf.doc_max_l])))
                   doc2id.append([unk_token(word2id, word) for word in doc_i[:conf.doc_max_l]]+[0]*(conf.doc_max_l-len(doc_i[:conf.doc_max_l])))
                   
                   #temp_seg=[i for i,v in enumerate(doc2id[-1]) if v==6]
                   #seg_mask.append(np.concatenate([np.ones(len(temp_seg)), np.zeros(conf.seg_delta-len(temp_seg))]))
                   #sent_seg.append(temp_seg+[0]*(conf.seg_delta-len(temp_seg)))
                   
                   query_len.append(len(que_i[:conf.que_max_l]))
                   sum_len.append(len(sum_i[:conf.sum_max_l]))
                    
                   query_mask.append([1]*len(que_i[:conf.que_max_l])+[0]*(conf.que_max_l-len(que_i[:conf.que_max_l])))
                   summ_mask.append([1]*len(sum_i[:conf.sum_max_l])+[0]*(conf.sum_max_l-len(sum_i[:conf.sum_max_l])))
                    
                   query2id.append([unk_token(word2id, word) for word in que_i[:conf.que_max_l]] + [0]*(conf.que_max_l-len(que_i[:conf.que_max_l])))
                   if inference:
                       summ2id.append([unk_token(word2id,     word) for word in sum_i[:conf.sum_max_l-1]] +[1]+ [0]*(conf.sum_max_l-len(sum_i[:conf.sum_max_l-1])-1))
                   else:
                       summ2id.append([unk_token(sum_word2id, word) for word in sum_i[:conf.sum_max_l-1]] +[1]+ [0]*(conf.sum_max_l-len(sum_i[:conf.sum_max_l-1])-1))
                       '''
                       copy_temp=[]
                       position_temp=[]
                       for word in sum_i[:len(sum_i[:conf.sum_max_l])-1]:
                           if word not in target_vocab:
                               copy_temp.append(1)
                               try:
                                   position_temp.append(doc_i.index(word))
                               except:
                                   position_temp.append(-1)
                                   
                           else:
                               copy_temp.append(0)
                               position_temp.append(-1)
                       copy_indicator.append(copy_temp + [1] + [0]*(conf.sum_max_l-len(sum_i[:conf.sum_max_l]))) 
                       position.append(position_temp + [1] + [0]*(conf.sum_max_l-len(sum_i[:conf.sum_max_l])))
                       '''
    return np.array(doc2id).astype('int32'),     np.array(query2id).astype('int32'),       np.array(summ2id).astype('int32'),    \
           np.array(doc_mask).astype('float32'), np.array(query_mask).astype('float32'),   np.array(summ_mask).astype('float32'),\
           np.array(doc_len).astype('int32'),    np.array(query_len).astype('int32'),      np.array(sum_len).astype('int32'), \
           np.array(sent_seg).astype('int32'),    np.array(seg_mask).astype('float32'),  \
           np.array(copy_indicator).astype('int32'), np.array(position).astype('int32')

cnn_train_data = convert_to_id(cnn_train_doc, cnn_train_query, cnn_train_summ, False)
cnn_val_data = convert_to_id(cnn_val_doc, cnn_val_query, cnn_val_summ, True)
cnn_test_data = convert_to_id(cnn_test_doc, cnn_test_query, cnn_test_summ, True)

daily_train_data = convert_to_id(daily_train_doc, daily_train_query, daily_train_summ, False)
daily_val_data = convert_to_id(daily_val_doc, daily_val_query, daily_val_summ, True)
daily_test_data = convert_to_id(daily_test_doc, daily_test_query, daily_test_summ, True)


def data_merge(data1, data2):
    data_sets=[]
    for data1_i, data2_i in zip(data1, data2):
        data_sets.append(np.concatenate((data1_i, data2_i), axis=0))
    return tuple(data_sets)

train_data = data_merge(cnn_train_data, daily_train_data)
val_data = data_merge(cnn_val_data, daily_val_data)
test_data = data_merge(cnn_test_data, daily_test_data)
# %%

print len(test_data）







'''
import gensim
from gensim.corpora.dictionary import Dictionary

def word2vec(emb_dim):
    model= gensim.models.KeyedVectors.load_word2vec_format("/Users/zy/Desktop/query_sum_only_cnn/glove.6B/glove.6B.%dd.txt"%emb_dim,binary=False)
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.vocab.keys(),allow_update=True)        
                
    glove_word2id = {word_: id_ for id_, word_ in gensim_dict.items()}          
    glove_word_vectors = {word: model[word] for word in glove_word2id.keys()}            
    
    embedding_weights = []
    for word in source_vocab:
        try:
            embedding_weights.append(glove_word_vectors[word])
        except:
            embedding_weights.append(np.random.normal(0.0, 0.5, emb_dim).astype('float32'))
    
    return np.array(embedding_weights).astype('float32')
            
            
            

def ava(doc1_train, doc1_val, doc1_test, doc2_train, doc2_val, doc2_test):
    lengths=0
    for d in doc1_train:
        lengths+=len(d)
        
    for d in doc1_val:
        lengths+=len(d)
    
    for d in doc1_test:
        lengths+=len(d)
    
    for d in doc2_train:
        lengths+=len(d)
        
    for d in doc2_val:
        lengths+=len(d)
    
    for d in doc2_test:
        lengths+=len(d)    
    
    
    
    
    nb=len(doc1_train)+len(doc1_val)+len(doc1_test)+len(doc2_train)+len(doc2_val)+len( doc2_test)
    return lengths/float(nb)      
            
print ava(daily_train_doc, daily_val_doc,daily_test_doc,cnn_train_doc,cnn_val_doc,cnn_test_doc)
print ava(daily_train_query, daily_val_query,daily_test_query,cnn_train_query,cnn_val_query,cnn_test_query)   
print ava(daily_train_summ, daily_val_summ,daily_test_summ,cnn_train_summ,cnn_val_summ,cnn_test_summ)          
'''    
            
            
            