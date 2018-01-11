# -*- coding: utf-8 -*-

class config(object):
    doc_max_l = 400
    que_max_l = 10
    sum_max_l = 20
    seg_delta = 16

    
    emb_dim = 200
    enco_hdim = 200
    query_hdim = enco_hdim
    deco_hdim = 2*enco_hdim
    source_vocab_size = 170000
    target_vocab_size = 50000
    batch_size = 20

    lr = 0.0005
    
