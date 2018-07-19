# -*- coding: utf-8 -*-
import numpy as np
import  tensorflow  as tf
import helpers
import pdb
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import pickle as pkl

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


tf.reset_default_graph()
sess = tf.InteractiveSession(config = tf.ConfigProto(allow_soft_placement=True))

#------------------
# パラメータの設定
PAD = 0
EOS = 1

vocab_size = 100
#input_embedding_size = 200
embedding_size = 200

encoder_hidden_units = 200
decoder_hidden_units = encoder_hidden_units

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
#encoderのcell作成及びdynamic_rnnの実行
with tf.variable_scope('encoder_fn') as scope:
    encoder_cell = CustomRNN(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        cell=encoder_cell,
        dtype=tf.float32,
        sequence_length=encoder_inputs_length,
        inputs=encoder_inputs_embedded,
    	time_major=True)
#encoder_dynamic_rnnの中間層の出力をoutputとstateに切り分ける
encoder_states = encoder_outputs[:,:,encoder_hidden_units:]
encoder_outputs = encoder_outputs[:,:,:encoder_hidden_units]


#------------------
# デコーダ
# decoder_inputs_embedded：デコーダの入力
decoder_cell = CustomRNN(decoder_hidden_units)

# decoderの文章の長さ
decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
decoder_lengths = encoder_inputs_length+1
#decoderのembedding行列作成
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
#decoderのcell作成及びdynamic_rnnの実行
with tf.variable_scope('decoder_fn') as scope:
    decoder_cell = CustomRNN(decoder_hidden_units)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    cell=decoder_cell,
    initial_state=encoder_final_state,
    dtype=tf.float32,
    sequence_length=decoder_lengths,
    inputs=decoder_inputs_embedded,
	time_major=True)
#decoder_dynamic_rnnの中間層の出力をoutputとstateに切り分ける
decoder_states = decoder_outputs[:,:,decoder_hidden_units:]
decoder_outputs = decoder_outputs[:,:,:decoder_hidden_units]

# decoder_outputsを、vocab_sizeのロジットに変換する線形変換の変数定義
W  =  tf.Variable(tf.random_uniform([1,decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
#luongのattentionで用いる変数定義
luong_W  =  tf.Variable(tf.random_uniform([1,decoder_hidden_units,encoder_hidden_units], -1, 1), dtype=tf.float32)
context_W = tf.Variable(tf.random_uniform([1,decoder_hidden_units,decoder_hidden_units*2], -1, 1), dtype=tf.float32)

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

#------------------
# 損失関数

# デコーダの教師出力
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


# 最大の単語IDを取得
decoder_prediction = tf.argmax(decoder_logits, 2)

labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)

# decoder_targetsとdecoder_predictionの交差エントロピー
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
	labels=labels,
	logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)
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

try:
	for batch in range(max_batches):
		fd = next_feed()
		l_W,luong_s,luong_weight,_, l = sess.run([luong_W,luong_score,luong_atW,train_op, loss], fd)
		loss_track.append(l)

		if batch == 0 or batch % batches_in_epoch == 0:
			print('batch {}'.format(batch))
			print('  minibatch loss: {}'.format(sess.run(loss, fd)))
			'''
			print('  attention_W:{}'.format(l_W))
			print('  attention_score:{}'.format(luong_s))
			print('  attention_weight:{}'.format(luong_weight))
			pdb.set_trace()
			'''
			predict_ = sess.run(decoder_prediction, fd)

			for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
				print('  sample {}:'.format(i + 1))
				print('	input	 > {}'.format(inp))
				print('	predicted > {}'.format(pred))
				if i >= 2:
					break
			print()

except KeyboardInterrupt:
	print('training interrupted')

plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))
plt.show()
