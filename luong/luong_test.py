# -*- coding: utf-8 -*-
import numpy as np
import  tensorflow  as tf
import helpers
import pdb
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import pickle as pkl
import pandas as pd
import os
import codecs


tf.reset_default_graph()
sess = tf.InteractiveSession(config = tf.ConfigProto(allow_soft_placement=True))
#lstmcellをoutput,stateで結合した形で中間層をすべて保存できるように設定(state_is_tuple)を変えるクラス
class CustomRNN(tf.contrib.rnn.LSTMCell):
    def __init__(self, *args, **kwargs):
        kwargs['state_is_tuple'] = False # force the use of a concatenated state.
        returns = super(CustomRNN, self).__init__(*args, **kwargs) # create an lstm cell
        self._output_size = self._state_size # change the output size to the state size
        return returns
    def __call__(self, inputs, state):
        output, next_state = super(CustomRNN, self).__call__(inputs, state)
        return next_state, next_state # return two copies of the state, instead of the output and the state


def weight_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name, shape):
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

def fc(inputs, w):
	fc = tf.matmul(inputs, w)
	return fc

#------------------
# パラメータの設定
PAD = 0
EOS = 1

vocab_size = 100
#input_embedding_size = 200
embedding_size = 1200

#h_dim = 200
#h_dim = h_dim
h_dim = 200


loss_track = []
max_batches = 10001
batches_in_epoch = 10
#------------------
# エンコーダとデコーダの入力
# 文章の長さ　×　バッチサイズ
encoder_inputs  = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
# 文章の長さ（encoder_max_time）とバッチサイズ（batch）をencoder_inputsから取得
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
# エンコーダの文章の長さ
# バッチサイズ
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
# 単語IDをembeddedベクトルに変換
# vocab_size × input_embedding_sizeの行列（ランダム）
embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), dtype=tf.float32)
# 文章の長さ　×　バッチサイズ　×　input_embedding_size（embeddedベクトルの次元）
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
# decoderの文章の長さ
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_lengths = encoder_inputs_length+1
#decoderのembedding行列作成
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

is_decoder_input = tf.placeholder(tf.bool, name='is_decoder_input')

#encoder
def encoder(x,x_length,test = False,keep_prob=0.5):
    with tf.variable_scope('encoder_fn') as scope:
        if test:
            scope.reuse_variables()

        cell = CustomRNN(h_dim)
        outputs_states, final_state = tf.nn.dynamic_rnn(
            cell = cell,
            dtype = tf.float32,
            sequence_length = x_length,
            inputs = x,
        	time_major=True)
    #encoder_dynamic_rnnの中間層の出力をoutputとstateに切り分ける
    states = outputs_states[:,:,:h_dim]
    outputs = outputs_states[:,:,h_dim:]

    return states,final_state

encoder_train_states,encoder_train_final_state = encoder(encoder_inputs_embedded,encoder_inputs_length,keep_prob=1)
encoder_test_states,encoder_test_final_state = encoder(encoder_inputs_embedded,encoder_inputs_length,test=True,keep_prob=1)

def decoder(encoder_final_state,encoder_states,test=False,keep_prob=0.5):
    with tf.variable_scope('decoder_fn') as scope:
        if test:
            scope.reuse_variables()
        #------------------
        # デコーダ
        # データ数の取得
        _, data_size, _ = tf.unstack(tf.shape(encoder_states))

        fcW = weight_variable("fcW", [1,h_dim, vocab_size])

        Wa = weight_variable("Wa", [1,h_dim,h_dim])
        Wc = weight_variable("Wc", [1,h_dim,h_dim*2])

		#--------------------
		# decoderの初期ステップ時の出力変換
        def loop_fn_initial(initial_state):
            initial_elements_finished = (0 >= encoder_inputs_length+1)  # all False at the initial step

            initial_input = tf.ones([data_size, embedding_size]) * EOS
            initial_cell_state = initial_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)
		#--------------------

		#--------------------
		# luong attention
        def get_luong_output(decoder_states):

			# time_majorから、データ数 x ステップ数 x h_dimに変換
            decoder_states_T = tf.transpose(decoder_states,[1,0,2])
            encoder_states_T = tf.transpose(encoder_states,[1,0,2])

			# パラメータの奥行をdata_sizeに拡張
            Was = tf.tile(Wa,[data_size,1,1])
            Wcs = tf.tile(Wc,[data_size,1,1])

			# スコアの計算（式8）
            scores = tf.matmul(tf.matmul(decoder_states_T,Was), tf.transpose(encoder_states,[1,2,0]))

			# align atの計算（式7）
			#aligns = tf.div(tf.exp(scores),tf.reduce_sum(tf.exp(scores),2,keep_dims=True)+0.01)
			# x-max(x):expのinfを回避するため
            scores_minus_max = scores-tf.reduce_max(scores,axis=2,keep_dims=True)
            aligns = tf.div(tf.exp(scores_minus_max),tf.reduce_sum(tf.exp(scores_minus_max),2,keep_dims=True)+0.01)

			# コンテキストベクターctの計算
            contexts = tf.matmul(aligns,encoder_states_T)

			# コンテキストベクターとdecoderの状態を結合（式5）
            context_decoder_states = tf.concat([tf.transpose(decoder_states_T,[0,2,1]),tf.transpose(contexts,[0,2,1])],1)

			# luong attentionの出力（式5）
            luong_outputs = tf.tanh(tf.matmul(Wcs,context_decoder_states))

            return luong_outputs
		#--------------------

		#--------------------
		# decoderの各ステップの出力変換
        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            def get_luong_output_loop():
                previous_output_luong = get_luong_output(tf.expand_dims(previous_output[:,:h_dim],axis=0))
                #previous_output_luong_info = tf.concat([previous_output_luong[:,:,0], info_inputs], axis=1)
                return fc(previous_output_luong,tf.transpose(fcW,perm=[2,0,1]))
                #return fc(previous_output_luong,fcW)

			# 一つ前のステップの出力previous_outputを線形変換に変換
            def get_next_input():
                pdb.set_trace()
                next_input = tf.stop_gradient(tf.cond(is_decoder_input,
                            lambda:decoder_inputs_embedded[time-1],
                            get_luong_output_loop)
                            )

                return next_input

			# 現在のステップtimeがencoder_inputs_length+1以上か否か
            elements_finished = (time >= encoder_inputs_length+1)

			# batch_sizeのelements_finishedをANDを計算。
            finished = tf.reduce_all(elements_finished)

			# 終了している場合、PAD、終了していない場合1つ前の出力を線形変換したものを次の入力にする
            pad_input = tf.ones([data_size, embedding_size]) * PAD
            input = tf.cond(finished, lambda:pad_input, get_next_input)
            state = previous_state		# hiddenはそのまま渡す
            output = previous_output	# outputもそのまま渡す
            loop_state = None			# loop_stateはNone

            return (elements_finished,
                    input,
                    state,
                    output,
                    loop_state)
		#--------------------

		#--------------------
		# 1ステップ目は、previous_outputがnoneのため、loop_fn_initial()を実行、2ステップ以降はloop_fn_transitionを実行
        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:	# time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial(encoder_final_state)
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

				# decoder_inputs：デコーダの入力
        # decoder_inputs_embedded：デコーダの入力
        cell = CustomRNN(h_dim)

        # 自作のloop_fnを用いたRNN
        outputs_states_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        outputs_states = outputs_states_ta.stack()

		# 出力をoutputとstateに切り分ける
        decoder_states = outputs_states[:,:,:h_dim]
        decoder_outputs = outputs_states[:,:,h_dim:]
		#----------------

		# luong attention
        outputs = get_luong_output(decoder_states)
		# outputs_flat = tf.reshape(outputs, (-1, h_dim))

		#----------------
		# outputsを、time_majorから、(データ数 x ステップ数) x h_dimに変換
        outputs_flat = tf.reshape(outputs, (-1, h_dim))
        outputs_flat_info = tf.concat([outputs_flat, tf.reshape(tf.tile(info_inputs,[1,encoder_inputs_length+1]),[-1,len(infoInds)])], axis=1)

		# fully connected network
        logits_flat = fc(outputs_flat_info,fcW)

		# time majorに戻す
        logits = tf.reshape(logits_flat, (encoder_inputs_length+1, data_size, embedding_size))
		#----------------
        return logits
#---------------------------

#---------------------------
# decoder
decoder_train_logits = decoder(encoder_train_final_state, encoder_train_states, keep_prob=1.0)
decoder_test_logits = decoder(encoder_test_final_state, encoder_test_states,test=True,keep_prob=1.0)
'''
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        cell=decoder_cell,
        initial_state=encoder_final_state,
        dtype=tf.float32,
        sequence_length=decoder_lengths,
        inputs=decoder_inputs_embedded,
    	time_major=True)
        #decoder_dynamic_rnnの中間層の出力をoutputとstateに切り分ける
        decoder_states = decoder_outputs[:,:,:h_dim]
        decoder_outputs = decoder_outputs[:,:,h_dim:]

# decoder_outputsを、vocab_sizeのロジットに変換する線形変換の変数定義
W  =  tf.Variable(tf.random_uniform([1,h_dim, vocab_size], -1, 1), dtype=tf.float32)
#luongのattentionで用いる変数定義
luong_W  =  tf.Variable(tf.random_uniform([1,h_dim,h_dim], -1, 1), dtype=tf.float32)
context_W = tf.Variable(tf.random_uniform([1,h_dim,h_dim*2], -1, 1), dtype=tf.float32)

# decoder_outputsの各次元の大きさ取得
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))

#Luong_Attentionを用いてdecoder_outputsの状態を入れ替える。
#encoder,decoderのstateの形を用いる式に合わせてtransposeする。
decoder_states_T = tf.transpose(decoder_states,perm=[1,0,2])
encoder_states_T = tf.transpose(encoder_states,perm=[1,0,2])
#作成しておいた3つのWのz軸をdecoderのz軸の長さに合わせる
luong_W = tf.tile(luong_W,[decoder_batch_size,1,1])
context_W = tf.tile(context_W,[decoder_batch_size,1,1])
W = tf.tile(W,[decoder_max_steps,1,1])

#luongのAttention
luong_score = tf.matmul(tf.matmul(decoder_states_T,luong_W),tf.transpose(encoder_states,perm=[1,2,0]))
luong_e = luong_score - tf.reduce_max(luong_score,2,keep_dims=True)
luong_atW = tf.div(tf.exp(luong_score),tf.reduce_sum(tf.exp(luong_score),2,keep_dims=True)+0.01)
luong_ct = tf.matmul(luong_atW,encoder_states_T)
luong_at_vec = tf.concat([tf.transpose(decoder_states_T,perm=[0,2,1]),tf.transpose(luong_ct,perm=[0,2,1])],1)
luong_result = tf.matmul(context_W,luong_at_vec)

# decoder_outputsを、一気にvocab_sizeのロジットに変換
# decoder_outputsを、(decoder_max_steps * decoder_batch_size) x docoder_dimにreshape
decoder_outputs_flat=tf.transpose(luong_result,[2,0,1])
decoder_logits_flat = tf.matmul(decoder_outputs_flat, W)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
#------------------
'''
#------------------
# 損失関数
# デコーダの教師出力
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


# 最大の単語IDを取得
decoder_prediction = tf.argmax(decoder_logits, 2)

labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)

# decoder_targetsとdecoder_predictionの交差エントロピー
stepwise_cross_entropy_train = tf.nn.softmax_cross_entropy_with_logits(
	labels=labels,
	logits=decoder_train_logits,
)
loss_train = tf.reduce_mean(stepwise_cross_entropy_train)

stepwise_cross_entropy_test = tf.nn.softmax_cross_entropy_with_logits(
	labels=labels,
	logits=decoder_test_logits,
)
loss_test = tf.reduce_mean(stepwise_cross_entropy_test)

#train_op = tf.train.AdamOptimizer().minimize(loss)
optimizer = tf.train.AdamOptimizer()
# 勾配のクリッピング
gvs = optimizer.compute_gradients(loss_train)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

#------------------

sess.run(tf.global_variables_initializer())

#------------------
# バッチデータのイテレータの作成
batch_size = 100
batches = helpers.random_sequences(length_from=3, length_to=10,
	   vocab_lower=2, vocab_upper=vocab_size,
	   batch_size=batch_size)
#pdb.set_trace()
#------------------

#------------------
# feed_dictの作成
def next_feed():
	batch = next(batches)

	# エンコーダの入力
	# 文章の最大長（length_to）　×　バッチサイズ（batch_size）
	encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)

	# デコーダの教師出力
	# 文章の最大長＋１（length_to）　×　バッチサイズ（batch_size）
	# エンコーダの入力の最後にEOS,[PAD]*2を付けたもの
	# [PAD]*2?
	decoder_targets_, _ = helpers.batch(
		[(sequence) + [EOS] for sequence in batch]
	)
	decoder_inputs_, _ = helpers.batch(
		[[EOS]+(sequence) for sequence in batch]
	)
	return{
		encoder_inputs: encoder_inputs_,
		encoder_inputs_length: encoder_input_lengths_,
		decoder_inputs: decoder_inputs_,
		decoder_targets: decoder_targets_,
	}
#------------------
#文章用(dualencoder 拡張用)
#------------------
def set_data(data_path,dict_path):
    #文章をidにするための辞書を読み込み
    dict = pd.read_csv(dict_path,header=None)
    #文章を読み込み
    fn_list = os.listdir(data_path)
    #del_str = ',.\n'
    f_str_list=[]
    for fn in fn_list:
        line_list = []
        fpath = os.path.join(data_path,fn)
        fp = codecs.open(fpath,'r','utf-8','ignore')
        for line in fp:
            line_sp = line.split(' ')
            line_sp = [st.strip(',.\n') for st in line_sp]
            line_sp = np.delete(np.array(line_sp),np.where(pd.DataFrame(line_sp)=='')[0]).tolist()
            line_list.append(line_sp)
        f_str_list.append(line_list)
        fp.close()

    return dict,f_str_list[0],f_str_list[1],f_str_list[2],f_str_list[3]


def next_feed_str(dict,in_text,out_text,max_length,index,count,trainmode=1):
    encoder_input_lengths_ = max_length
    encoder_inputs_text = in_text[batch_size*count:batch_size*(count+1)]
    decoder_inputs_text = out_text[batch_size*count:batch_size*(count+1)]

    encoder_lengths = np.array([len(t) for t in encoder_inputs_text])

    encoder_inputs_ = []
    for sentence in encoder_inputs_text:
        line_ids = []
        for t in range(encoder_input_lengths_):
            if len(sentence) <= t:
                line_ids.extend([0])
            else:
                line_ids.extend(np.where(dict == sentence[t])[0].tolist())
        encoder_inputs_.append(line_ids)

    encoder_inputs_array = np.array(encoder_inputs_).transpose()

    decoder_inputs_ = []
    decoder_targets_ = []
    for sentence in decoder_inputs_text:
        line_ids = []
        line_ids_reverse = []
        for t in range(encoder_input_lengths_+1):
            if len(sentence) == t:
                line_ids.extend([1])
                line_ids_reverse = line_ids[::-1]
            elif len(sentence) <= t:
                line_ids.extend([0])
                line_ids_reverse.extend([0])
            else:
                line_ids.extend(np.where(dict == sentence[t])[0].tolist())
        decoder_inputs_.append(line_ids_reverse)
        decoder_targets_.append(line_ids)

    decoder_inputs_array = np.array(decoder_inputs_).transpose()
    decoder_targets_array = np.array(decoder_targets_).transpose()
    count = count + 1

    if batch_size*(count+1) > len(in_text):
        index = np.random.permutation(len(in_text))

    if trainMode == 1:
        is_decoder_input_ = True
    else :
        is_decoder_input_ = False

    return{
        encoder_inputs:encoder_inputs_array,
        encoder_inputs_length:encoder_lengths,
        decoder_inputs:decoder_inputs_array,
        decoder_targets:decoder_targets_array,
        is_decoder_input: is_decoder_input_
    },index,count


dict,train_out,train_in,test_out,test_in = set_data('data/cornell_corpus','data/dict.csv')

train_randIndex = np.random.permutation(len(train_in))
test_randIndex = np.random.permutation(len(test_in))
train_lengths = np.max([np.max([len(t) for t in train_in]),np.max([len(t) for t in train_out])])
test_lengths = np.max([np.max([len(t) for t in test_in]),np.max([len(t) for t in test_out])])
train_batchcount = 0
test_batchcount = 0
vocab_size = dict.size

try:
    for batch in range(max_batches):
        if batch <= 5000:fd_train,train_randIndex,train_batchcount = next_feed_str(dict,train_in,train_out,train_lengths,train_randIndex,train_batchcount)
        else:fd_train,train_randIndex,train_batchcount = next_feed_str(dict,train_in,train_out,train_lengths,train_randIndex,train_batchcount,trainMode=2)

        _,loss_train_value,decoder_train_logits_values = sess.run([train_op, loss_train, decoder_train_logits], fd_train)

        if batch == 0 or batch % batches_in_epoch == 0:
            fd_test = next_feed_strnext_feed_str(dict,test_in,test_out,test_lengths,test_randIndex,test_batchcount,trainMode=2)
            loss_test_value, decoder_test_logits_values = sess.run([loss_test, decoder_test_logits], fd_test)

            print('batch {}'.format(batch))
            print('train loss: {}'.format(loss_train_value))
            print('test loss: {}'.format(loss_test_value))
            predict_ = sess.run(decoder_prediction, fd_test)

            for i, (inp, pred) in enumerate(zip(fd_test[encoder_inputs].T, predict_.T)):
                #pdb.set_trace()
                print('  sample {}:'.format(i + 1))
                inp_str = np.array([dict[0][ind] for ind in inp.tolist() if ind > 1])
                print('	input	 > {}'.format(inp_str))
                pred_str = np.array([dict[0][ind] for ind in pred.tolist() if ind > 1])
                print('	predicted > {}'.format(pred_str))
                if i >= 0:
                    break
            print()

except KeyboardInterrupt:
	print('training interrupted')

#plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
#plt.show()
