# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
'''

import tensorflow as tf
import numpy as np
import config
conf = config.config()

#data preparation
import data_process

train_doc2id,   train_query2id,   train_summ2id,  \
train_doc_mask, train_query_mask, train_sum_mask, \
train_doc_len,  train_que_len,    train_sum_len,  \
train_sent_seg, train_seg_mask, \
_, _= data_process.train_data

test_doc2id,   test_query2id,   test_summ2id,  \
test_doc_mask, test_query_mask, test_sum_mask, \
test_doc_len,  test_que_len,    test_sum_len,  \
test_sent_seg, test_seg_mask, \
_, _= data_process.test_data


# %%
#1 input
encoder_inputs  = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
query_inputs    = tf.placeholder(shape=(None, None), dtype=tf.int32, name='query_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
sent_seg        = tf.placeholder(shape=(None, None), dtype=tf.int32, name='sent_seg')
index_tf        = tf.placeholder(shape=(None,), dtype=tf.int32, name='index_tf')

encoder_inputs_length  = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
query_inputs_length    = tf.placeholder(shape=(None,), dtype=tf.int32, name='query_inputs_length')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets_length')

sum_mask_tf = tf.placeholder(shape=(None, None), dtype=tf.float32, name='sum_mask')
doc_mask_tf = tf.placeholder(shape=(None, None), dtype=tf.float32, name='doc_mask')
que_mask_tf = tf.placeholder(shape=(None, None), dtype=tf.float32, name='que_mask')
seg_mask_tf = tf.placeholder(shape=(None, None), dtype=tf.float32, name='seg_mask')

#2 embedding
embedding_weights = data_process.word2vec(conf.emb_dim)
with tf.device('/cpu:0'):
    embeddings = tf.Variable(tf.convert_to_tensor(embedding_weights), dtype=tf.float32, name='embeddings')
    '''
    emb_weight = tf.Variable(tf.constant(0.0, shape=[conf.source_vocab_size, conf.emb_dim]),
                    trainable=True, name="emb_weight")
    embedding_placeholder = tf.placeholder(tf.float32, [conf.source_vocab_size, conf.emb_dim])
    embeddings = emb_weight.assign(embedding_placeholder)
    '''
    #embeddings = tf.Variable(tf.truncated_normal([source_vocab_size, conf.emb_dim], mean=0.0, stddev=0.1), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    query_inputs_embedded   = tf.nn.embedding_lookup(embeddings, query_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_targets)

#3.1 encoder_doc
with tf.variable_scope('encoder_doc_fw1'):
    encoder_cell_fw1 = tf.contrib.rnn.GRUCell(conf.enco_hdim, kernel_initializer=tf.orthogonal_initializer())

with tf.variable_scope('encoder_doc_bw1'):
    encoder_cell_bw1 = tf.contrib.rnn.GRUCell(conf.enco_hdim, kernel_initializer=tf.orthogonal_initializer())

with tf.variable_scope('word_rnn'):
    ((encoder_fw_outputs,
      encoder_bw_outputs),
     (encoder_fw_final_state,
      encoder_bw_final_state))  = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw1,
                                            cell_bw=encoder_cell_bw1,
                                            inputs=encoder_inputs_embedded,
                                            sequence_length=encoder_inputs_length,
                                            dtype=tf.float32)
    
encoder_hiddens     = tf.concat((encoder_fw_outputs,     encoder_bw_outputs),    2)#batch_size, oc_max_len, decoder_hidden_units
encoder_final_state = tf.concat((encoder_fw_final_state, encoder_bw_final_state),1)

#3.2 sent_encoder
with tf.variable_scope('sent_cell_fw'):
    sent_cell_fw = tf.contrib.rnn.GRUCell(conf.enco_hdim, kernel_initializer=tf.orthogonal_initializer())

with tf.variable_scope('sent_cell_bw'):
    sent_cell_bw = tf.contrib.rnn.GRUCell(conf.enco_hdim, kernel_initializer=tf.orthogonal_initializer())
    
with tf.variable_scope('sent_rnn'):
    word_mask = []
    sent_mask = []
    length_segs = []
    sent_hiddens_seqs = []
    sent_states = []
    for seg in range(0, conf.doc_max_l, conf.seg_delta):
        word_mask.append(doc_mask_tf[:, seg:seg+conf.seg_delta])
        sent_mask.append(tf.reduce_prod(doc_mask_tf[:, seg:seg+conf.seg_delta], 1))
        length_seg=tf.cast(tf.reduce_sum(doc_mask_tf[:, seg:seg+conf.seg_delta], 1), tf.int32)
        length_segs.append(length_seg)
        ((sent_fw_outputs,
          sent_bw_outputs),
         (sent_fw_final_state,
          sent_bw_final_state))  = tf.nn.bidirectional_dynamic_rnn(cell_fw=sent_cell_fw,
                                            cell_bw=sent_cell_bw,
                                            inputs=encoder_hiddens[:, seg:seg+conf.seg_delta, :],
                                            sequence_length=length_seg,
                                            dtype=tf.float32)
         
        sent_hiddens     = tf.concat((sent_fw_outputs, sent_bw_outputs),         2) #batch_size, oc_max_len, decoder_hidden_units
        sent_final_state = tf.concat((sent_fw_final_state, sent_bw_final_state), 1)
        
        sent_hiddens_seqs.append(sent_hiddens)
        sent_states.append(sent_final_state)
        
word_mask         = tf.convert_to_tensor(word_mask)
sent_mask         = tf.transpose(tf.convert_to_tensor(sent_mask), [1, 0])
sent_hiddens_seqs = tf.convert_to_tensor(sent_hiddens_seqs)
sent_states       = tf.transpose(tf.convert_to_tensor(sent_states), [1, 0, 2])
length_segs       = tf.transpose(tf.convert_to_tensor(length_segs), [1, 0])



#4 encoder_query
with tf.variable_scope('encoder_query'):
    query_cell = tf.contrib.rnn.GRUCell(conf.query_hdim, kernel_initializer=tf.orthogonal_initializer())
    query_hiddens, query_final_state = tf.nn.dynamic_rnn(cell=query_cell,                                       
                                                         inputs=query_inputs_embedded,
                                                         sequence_length=query_inputs_length, 
                                                         dtype=tf.float32)

#4 decoder

with tf.variable_scope('decoder1'):
    cell = tf.contrib.rnn.GRUCell(conf.deco_hdim, kernel_initializer=tf.orthogonal_initializer())

start_time_slice = tf.constant(2, shape=[conf.batch_size], dtype=tf.int32, name='start')
start_step_embedded = tf.nn.embedding_lookup(embeddings, start_time_slice)

w_proj = tf.Variable(tf.truncated_normal([conf.emb_dim, 2*conf.deco_hdim+conf.query_hdim], 0, 0.1), dtype=tf.float32, name='w_proj')
w_softmax = tf.tanh(tf.matmul(embeddings[:conf.target_vocab_size], w_proj))
w_softmax=tf.transpose(w_softmax, [1,0])

w_doc = tf.Variable(tf.truncated_normal([conf.deco_hdim+2*conf.enco_hdim+conf.query_hdim, conf.deco_hdim], 0, 0.1), dtype=tf.float32,name='w_doc')
v_doc = tf.Variable(tf.truncated_normal([conf.deco_hdim, 1], 0, 0.1), dtype=tf.float32,name='v_doc')

w_que = tf.Variable(tf.truncated_normal([conf.deco_hdim+conf.query_hdim, conf.deco_hdim], 0, 0.1), dtype=tf.float32,name='w_que')
v_que = tf.Variable(tf.truncated_normal([conf.deco_hdim, 1], 0, 0.1), dtype=tf.float32,name='v_que')

w_sent = tf.Variable(tf.truncated_normal([conf.deco_hdim+conf.deco_hdim, conf.deco_hdim], 0, 0.1), dtype=tf.float32,name='w_sent')
v_sent = tf.Variable(tf.truncated_normal([conf.deco_hdim, 1], 0, 0.1), dtype=tf.float32,name='v_sent')

#%%

def attention(st, w, v, enco_hs, mask, seq_l, query_context=None):
    #w_repeat: [64, 400, 200], v_repeat: [64, 200, 1]
    w_repeat = tf.reshape(tf.tile(w, [conf.batch_size, 1]), [conf.batch_size, w.shape.as_list()[0], -1])
    v_repeat = tf.reshape(tf.tile(v, [conf.batch_size, 1]), [conf.batch_size, -1, 1])
    st_repeat = tf.reshape(tf.tile(st, [1, seq_l]), [conf.batch_size, seq_l, -1])#[64, 120, 200]
    
    if query_context!=None:
        query_hiddens_repeat = tf.reshape(tf.tile(query_context, [1, seq_l]), [conf.batch_size, seq_l, -1])
        cat_hiddens = tf.concat([st_repeat, enco_hs, query_hiddens_repeat], 2)#[64, 120, 400]
    else:
        cat_hiddens = tf.concat([st_repeat, enco_hs], 2)
    
    temp = tf.matmul(cat_hiddens, w_repeat)#[64, 120, 200]
    score_doc = tf.matmul(tf.tanh(temp), v_repeat)#[64, 120, 1]
    score_doc = tf.reshape(score_doc, [conf.batch_size, -1])#[64, 120]
    attention_doc = tf.nn.softmax(tf.add(score_doc, -100000*(1-mask)))#[64, 120]
    attention_doc = tf.reshape(attention_doc, [conf.batch_size, -1, 1])#[64, 120, 1]
    context = tf.reduce_sum(attention_doc*enco_hs, 1)#[64, 200]
    return context, attention_doc[:,:,0]


outputs=[]
states=[]
sent_attentions=[]
word_attentions=[]

is_training = tf.placeholder(tf.bool)

with tf.variable_scope('time_step'):
    for time_step in range(conf.sum_max_l):
        if time_step > 0:
            tf.get_variable_scope().reuse_variables()
        
        if time_step == 0:
            query_context, _ = attention(encoder_final_state, 
                                      w_que, 
                                      v_que, 
                                      query_hiddens, 
                                      que_mask_tf, 
                                      conf.que_max_l)
            
            
            word_atts=[]
            sent_states = []
            for s in range(conf.doc_max_l/conf.seg_delta):
                sent_context, word_att = attention(encoder_final_state, 
                                         w_sent, 
                                         v_sent, 
                                         sent_hiddens_seqs[s], 
                                         word_mask[s], 
                                         conf.seg_delta)
                
                sent_states.append(sent_context)
                word_atts.append(word_att)
            
            sent_states = tf.transpose(tf.convert_to_tensor(sent_states), [1, 0, 2])
            
            #sent_attention [20, 64, 25]
            att_context, sent_att = attention(encoder_final_state, 
                                      w_doc, 
                                      v_doc, 
                                      sent_states, 
                                      sent_mask, 
                                      conf.doc_max_l/conf.seg_delta,
                                      query_context)
            
            sent_attentions.append(sent_att)
            
            output, state = cell(tf.concat([start_step_embedded, att_context], 1), encoder_final_state)
            #penu_c = tf.matmul(output, w1_penu)+tf.matmul(att_context, w2_penu)+tf.matmul(query_final_state, w3_penu)
            penu_c = tf.concat([output, att_context, query_context], 1)
            #penu_c = tf.concat([output, att_context], 1)
            
            output_logits = tf.matmul(penu_c, w_softmax)
            prediction = tf.argmax(output_logits, axis=1)
            next_input = tf.nn.embedding_lookup(embeddings, prediction)
        
        else:
            query_context, _= attention(states[-1], 
                                      w_que, 
                                      v_que, 
                                      query_hiddens, 
                                      que_mask_tf,
                                      conf.que_max_l)
                                      
            word_atts=[]
            sent_states = []
            for s in range(conf.doc_max_l/conf.seg_delta):
                sent_context, word_att = attention(states[-1], 
                                         w_sent, 
                                         v_sent, 
                                         sent_hiddens_seqs[s], 
                                         word_mask[s],
                                         conf.seg_delta)
                sent_states.append(sent_context)
                word_atts.append(word_att)
            sent_states = tf.transpose(tf.convert_to_tensor(sent_states), [1, 0, 2])
            
            att_context ,sent_att = attention(states[-1], 
                                      w_doc, 
                                      v_doc, 
                                      sent_states, 
                                      sent_mask, 
                                      conf.doc_max_l/conf.seg_delta,
                                      query_context)
            output, state = cell(tf.cond(is_training, lambda: tf.concat([decoder_inputs_embedded[:,time_step-1,:], att_context], 1), \
                                                      lambda: tf.concat([next_input,                               att_context], 1)),\
                                 states[-1])
            
            sent_attentions.append(sent_att)
            #penu_c = tf.matmul(output, w1_penu)+tf.matmul(att_context, w2_penu)+tf.matmul(query_final_state, w3_penu)
           
            penu_c = tf.concat([output, att_context, query_context], 1)
            #penu_c = tf.concat([output, att_context], 1)
            
            output_logits = tf.matmul(penu_c, w_softmax)
            prediction = tf.argmax(output_logits, axis=1)
            next_input = tf.nn.embedding_lookup(embeddings, prediction)
            
            mask_slice = sum_mask_tf[:,time_step]
            mask_slice = tf.expand_dims(mask_slice, -1)
            
            state = state*mask_slice + states[-1]*(1.0-mask_slice)
            
            '''
            state_c = state[0]*mask_slice + states[-1][0]*(1.0-mask_slice)
            state_h = state[1]*mask_slice + states[-1][1]*(1.0-mask_slice)
            state = tf.contrib.rnn.LSTMStateTuple(c=state_c, h=state_h)
            '''
            
        outputs.append(output_logits)
        states.append(state)
        
        word_attentions.append(word_atts)
        
    sent_attentions = tf.transpose(tf.convert_to_tensor(sent_attentions), [1, 0, 2])
    word_attentions = tf.transpose(tf.convert_to_tensor(word_attentions), [2, 0, 1, 3])
    
    decoder_logits = tf.convert_to_tensor(outputs)
    decoder_logits = tf.transpose(decoder_logits, [1, 0, 2])
    decoder_prediction = tf.argmax(decoder_logits, 2) 
    y_true=tf.one_hot(decoder_targets, depth=conf.target_vocab_size, dtype=tf.float32)


#%%
#vocab_size
#5 Optimizer
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=decoder_logits)

loss = tf.div(tf.reduce_sum(stepwise_cross_entropy*sum_mask_tf), tf.reduce_sum(sum_mask_tf))

tf.summary.scalar('loss',loss)
train_op = tf.train.AdamOptimizer(conf.lr).minimize(loss)
#%%




#session
loss_track = []   
epoches = 1000
total = len(train_doc2id)
batch_size = conf.batch_size
iterations = total/batch_size

nb_batch = len(test_doc2id)/conf.batch_size

def save_attention(session, string, rouge_s):
    for m in range(0, nb_batch*conf.batch_size, conf.batch_size):
        w_attention, s_attention = session.run([word_attentions, sent_attentions],
                     feed_dict={encoder_inputs        : test_doc2id[m:m+conf.batch_size],
                                query_inputs          : test_query2id[m:m+conf.batch_size],
                                decoder_targets       : test_summ2id[m:m+conf.batch_size],
                                encoder_inputs_length : test_doc_len[m:m+conf.batch_size],
                                query_inputs_length   : test_que_len[m:m+conf.batch_size],
                                decoder_targets_length: test_sum_len[m:m+conf.batch_size],
                                sum_mask_tf           : test_sum_mask[m:m+conf.batch_size],
                                doc_mask_tf           : test_doc_mask[m:m+conf.batch_size],
                                que_mask_tf           : test_query_mask[m:m+conf.batch_size],
                                #embedding_placeholder : embedding_weights,
                                is_training           : False,
                               })
        f = open('./model_att/'+string+'.txt','w')
        f.write("R1_%f_R2_%f_RL_%f"%(rouge_s[0], rouge_s[1], rouge_s[2]))
        f.close()
        np.save("./model_att/"+string+"word_att_%d_%d"%(m, m+conf.batch_size), w_attention)
        np.save("./model_att/"+string+"sent_att_%d_%d"%(m, m+conf.batch_size), s_attention)



import rouge
def rouge_score(session):
    assert nb_batch*conf.batch_size%conf.batch_size==0
    pred_sum=[]
    for m in range(0, nb_batch*conf.batch_size, conf.batch_size):
        pred = session.run(decoder_prediction,
                 feed_dict={encoder_inputs        : test_doc2id[m:m+conf.batch_size],
                            query_inputs          : test_query2id[m:m+conf.batch_size],
                            decoder_targets       : test_summ2id[m:m+conf.batch_size],
                            encoder_inputs_length : test_doc_len[m:m+conf.batch_size],
                            query_inputs_length   : test_que_len[m:m+conf.batch_size],
                            decoder_targets_length: test_sum_len[m:m+conf.batch_size],
                            sum_mask_tf           : test_sum_mask[m:m+conf.batch_size],
                            doc_mask_tf           : test_doc_mask[m:m+conf.batch_size],
                            que_mask_tf           : test_query_mask[m:m+conf.batch_size],
                            #embedding_placeholder : embedding_weights,
                            is_training           : False,
                           })
    
        pred_sum.extend(pred.tolist())
    
    assert len(pred_sum)==nb_batch*conf.batch_size
    rouge1_sum=[]
    rouge2_sum=[]
    rougel_sum=[]
    for i in range(nb_batch*conf.batch_size):
        pred_temp=[]
        ref_temp=[]
        for id_ in pred_sum[i]:
            if id_==1: break
            pred_temp.append(str(id_))
        
        for id_ in test_summ2id[i]:
            if id_==1: break
            ref_temp.append(str(id_))
        
        if pred_temp==[] or ref_temp==[]:
            continue
        
        rouge1_sum.append(rouge.rouge_n(pred_temp, ref_temp, n=1)[-1])
        rouge2_sum.append(rouge.rouge_n(pred_temp, ref_temp, n=2)[-1])
        rougel_sum.append(rouge.rouge_l(pred_temp, ref_temp))
        
     
    #print "rouge_1:,rouge1_sum/float(split))
    #print "rouge_2:%f"%(rouge2_sum/float(split))
    #print "rouge_l:%f"%(rougel_sum/float(split))
    return np.mean(rouge1_sum), np.mean(rouge2_sum), np.mean(rougel_sum), \
           np.std(rouge1_sum) , np.std(rouge2_sum), np.std(rougel_sum), pred_sum
  
    
    
    
    
    
    
id2word = data_process.id2word

def pint(pred, gold, is_print=False, is_write=False, name=None, rouge_s=None):
    
    pred_sum=[]
    gold_sum=[]
    
    for pred_sentence, golds in zip(pred, gold):
        pred_temp=[]
        gold_temp=[]
        for id_ in pred_sentence:
            if (id2word[id_]=='<eos>'):
                break
            pred_temp.append(id2word[id_])
          
        for id_ in golds:
            if (id2word[id_]=='<eos>') or (id2word[id_]=='<pad>') :
                break
            gold_temp.append(id2word[id_])
        
        pred_sum.append(" ".join(pred_temp).encode('utf-8'))
        gold_sum.append(" ".join(gold_temp).encode('utf-8'))
        
    if is_print:
        num=0
        for g, p in zip(gold_sum, pred_sum ):
            
            print "gold%d:"%num + g
            print "s%d:"%num + p +'\n'
            num+=1
    
    if is_write:
        f_res = open('/Users/zy/Desktop/ecir/debate/model_att/result'+name+'_R1_%f_R2_%f_RL_%f_R1std_%f_R2std_%f_RLstd_%f'%(rouge_s[0], rouge_s[1], rouge_s[2],rouge_s[3], rouge_s[4], rouge_s[5])+'.txt','w')
        #f_res.write('R1_%f_R2_%f_RL_%f'%(rouge_s[0], rouge_s[1], rouge_s[2])+'\n')
        num=0
        for g, p in zip(gold_sum, pred_sum):
            f_res.write("gold%d: "%num+g+'\n'+"pred%d: "%num+p+'\n\n')
            num+=1
        f_res.close()
        
    return pred_temp, gold_temp


import random
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  
    merged = tf.summary.merge_all() 
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    r1_max=-1000
    r2_max=-1000
    for epoch in range(epoches):
        
        random_tuple = zip(train_doc2id, train_query2id, train_summ2id, train_doc_len, train_que_len, \
                           train_sum_len, train_sum_mask, train_doc_mask, train_query_mask)
        random.shuffle(random_tuple)
        train_docs2id, train_query2id, train_summ2id, train_doc_len, train_que_len, \
        train_sum_len, train_sum_mask, train_doc_mask, train_query_mask = zip(*random_tuple)
        
        for k in range(0, iterations*batch_size, batch_size):
            res=sess.run([train_op, loss, merged], 
                         feed_dict={encoder_inputs        : train_docs2id[k:k+batch_size],
                                    query_inputs          : train_query2id[k:k+batch_size],
                                    decoder_targets       : train_summ2id[k:k+batch_size],
                                    encoder_inputs_length : train_doc_len[k:k+batch_size],
                                    query_inputs_length   : train_que_len[k:k+batch_size],
                                    decoder_targets_length: train_sum_len[k:k+batch_size],
                                    sum_mask_tf           : train_sum_mask[k:k+batch_size],
                                    doc_mask_tf           : train_doc_mask[k:k+batch_size],
                                    que_mask_tf           : train_query_mask[k:k+batch_size],
                                    #embedding_placeholder : embedding_weights,
                                    is_training           : True,
                                    })
            
            loss_track.append(res[1])
            writer.add_summary(res[2], k)
            print res[1]
            #print sess.run(tf.trainable_variables()[0])
            
            if k/batch_size%==0:
                #is_trainng=False
                
                pred=sess.run(decoder_prediction, 
                              feed_dict={encoder_inputs        : val_doc2id[:batch_size],
                                         query_inputs          : val_query2id[:batch_size],
                                         decoder_targets       : val_summ2id[:batch_size],
                                         encoder_inputs_length : val_doc_len[:batch_size],
                                         query_inputs_length   : val_que_len[:batch_size],
                                         decoder_targets_length: val_sum_len[:batch_size],
                                         sum_mask_tf           : val_sum_mask[:batch_size],
                                         doc_mask_tf           : val_doc_mask[:batch_size],
                                         que_mask_tf           : val_query_mask[:batch_size],
                                         #embedding_placeholder : embedding_weights,
                                         is_training           : False,
                                         })
               
                
                
               
                
                pint(pred[:batch_size], test_summ2id[:batch_size], is_print=True)
                
            
                print "epoch%d, iteration%d"%(epoch, k)
                rouge_tuple = rouge_score(sess)
                print "r1, r2, rl", rouge_tuple[:3]
                
                if rouge_tuple[0]>r1_max:
                    r1_max=rouge_tuple[0]
                    if r1_max>0.15:
                        model_saver = tf.train.Saver()
                        model_saver.save(sess,"model_best/model_r1_%f_r2_%f_rl_%f"%(rouge_tuple[0], rouge_tuple[1], rouge_tuple[2]))
                        save_attention(sess, "r1", rouge_tuple)
                        
                    pint(rouge_tuple[-1], test_summ2id, is_write=True, name='r1', rouge_s=rouge_tuple)
                    
               
                if rouge_tuple[1]>r2_max:
                    r2_max=rouge_tuple[1]
                    if r2_max>0.04:
                        model_saver = tf.train.Saver()
                        model_saver.save(sess,"model_best/model_r2_%f_r1_%f_rl_%f"%(rouge_tuple[1], rouge_tuple[0], rouge_tuple[2]))
                        save_attention(sess, "r2", rouge_tuple)
                    pint(rouge_tuple[-1], test_summ2id, is_write=True, name='r2', rouge_s=rouge_tuple)    
                    
                 
                
    writer.close()












