# -*- coding: utf-8 -*-
import numpy as np
import  tensorflow  as tf
import helpers
import pdb
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

tf.reset_default_graph()
sess = tf.InteractiveSession(config = tf.ConfigProto(allow_soft_placement=True))

#------------------
# パラメータの設定
PAD = 0
EOS = 1

vocab_size = 100
input_embedding_size = 1000

encoder_hidden_units = 1000
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
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

# 文章の長さ　×　バッチサイズ　×　input_embedding_size（embeddedベクトルの次元）
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

encoder_inputs_ta = tf.TensorArray(dtype=tf.float32, size=encoder_max_time)
encoder_inputs_ta = encoder_inputs_ta.unstack(encoder_inputs_embedded)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

def encoder_loop_fn(time, cell_output, cell_state, loop_state):
	emit_output = cell_output  # == None for time == 0
	if cell_output is None:  # time == 0
		next_cell_state = encoder_cell.zero_state(batch_size, tf.float32)
		#encoder_hidden = tf.expand_dims(next_cell_state,0)
		#pdb.set_trace()
	else:
		next_cell_state = cell_state

	elements_finished = (time >= encoder_inputs_length)
	finished = tf.reduce_all(elements_finished)
	next_input = tf.cond(
		finished,
	    lambda: tf.zeros([batch_size, input_embedding_size], dtype=tf.float32),
	    lambda: encoder_inputs_ta.read(time))
	next_loop_state = None
	return (elements_finished, next_input, next_cell_state,
	        emit_output, next_loop_state)
with tf.variable_scope('encoder_fn') as scope:
	encoder_outputs_ta, encoder_final_state, _ = tf.nn.raw_rnn(encoder_cell, encoder_loop_fn)
	encoder_outputs = encoder_outputs_ta.stack()
#------------------
# デコーダ
# decoder_inputs_embedded：デコーダの入力（学習時のみ、テストのときはどうする？）
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
#decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)
#decoder_cell = tf.nn.rnn_cell.MultiRNNCell([decoder_cell]*2)

# decoderの文章の長さ
# # +2 additional steps, +1 leading <EOS> token for decoder inputs
decoder_lengths = encoder_inputs_length + 3

#decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
#	decoder_cell, decoder_inputs_embedded,
#	initial_state=encoder_final_state,
#	dtype=tf.float32, time_major=True, scope="plain_decoder",
#)

# decoder_outputsを、vocab_sizeのロジットに変換する線形変換の変数
#decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
W  =  tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

assert EOS == 1 and PAD == 0

# EOSとPADのembedded vectorを用意。batch_size分
eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

# decoderの初期ステップ時の出力変換
def loop_fn_initial():
	initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
	initial_input = eos_step_embedded
	initial_cell_state = encoder_final_state
	initial_cell_output = None
	initial_loop_state = None  # we don't need to pass any additional information
	return (initial_elements_finished,
			initial_input,
			initial_cell_state,
			initial_cell_output,
			initial_loop_state)

# decoderの各ステップの出力変換
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

	# 一つ前のステップの出力をembedded vectorに変換
	def get_next_input():
		# previous_outputを、vocab_sizeのロジットに変換
		output_logits = tf.add(tf.matmul(previous_output, W), b)
		prediction = tf.argmax(output_logits, axis=1)
		next_input = tf.nn.embedding_lookup(embeddings, prediction)
		return next_input

	# 現在のステップtimeがdecoder_lengths以上か否か、decoder_lengthsはbatch_size次元
	elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
												  # defining if corresponding sequence has ended

	# batch_sizeのelements_finishedをスカラーに落としている。１要素でもFalseがあれば、Falseになる。これでいいの？
	finished = tf.reduce_all(elements_finished) # -> boolean scalar
	input = tf.cond(finished, lambda: pad_step_embedded, get_next_input) # 終了している場合、PAD、終了していない場合一つまえの出力を返す
	state = previous_state		# hiddenはそのまま渡す
	output = previous_output	# outputもそのまま渡す
	loop_state = None			# loop_stateはNone

	return (elements_finished,
			input,
			state,
			output,
			loop_state)

# 1ステップ目は、previous_outputがnoneのため、loop_fn_initial()を実行、2ステップ以降はloop_fn_transitionを実行
def loop_fn(time, previous_output, previous_state, previous_loop_state):
	#pdb.set_trace()
	if previous_state is None:	# time == 0
		assert previous_output is None and previous_state is None
		return loop_fn_initial()
	else:
		return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

# 自作の出力入力変換loop_fnを用いて、rnnし、decoder_max_steps x decoder_batch_size x decoder_dimの出力を獲得する
with tf.variable_scope('decoder_fn') as scope:
	decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
	decoder_outputs = decoder_outputs_ta.stack()
#------------------

#------------------
# decoder_outputsを、一気にvocab_sizeのロジットに変換

# decoder_outputsの各次元の大きさ取得
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))

# decoder_outputsを、(decoder_max_steps * decoder_batch_size) x docoder_dimにreshape
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
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
batches = helpers.random_sequences(length_from=30, length_to=80,
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
		[(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
	)

	return {
		encoder_inputs: encoder_inputs_,
		encoder_inputs_length: encoder_input_lengths_,
		decoder_targets: decoder_targets_,
	}
#------------------

try:
	for batch in range(max_batches):
		fd = next_feed()
		_, l = sess.run([train_op, loss], fd)
		loss_track.append(l)

		if batch == 0 or batch % batches_in_epoch == 0:
			print('batch {}'.format(batch))
			print('  minibatch loss: {}'.format(sess.run(loss, fd)))

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
