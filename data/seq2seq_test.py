import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.layers import core as layers_core
import pdb

class Seq2Seq_tf:
    def __init__(self,hparams,sos_id,eos_id,padding_id):
       	#ハイパーパラメータ
       	self.hparams = hparams
        #文頭と文末を示すID
        self.tgt_sos_id = sos_id
        self.tgt_eos_id = eos_id
        self.padding_id = padding_id
        self.batchcnt = 0

    def set_seq2seq(self):
        #Encoder
        self.encoder_inputs = tf.placeholder(tf.int32, shape=(self.hparams.encoder_length,self.hparams.batch_size),name="encoder_inputs")

        embedding_encoder = tf.get_variable("embedding_encoder",[self.hparams.src_vocab_size,self.hparams.embedding_size])

        encoder_emb_inputs = tf.nn.embedding_lookup(embedding_encoder,self.encoder_inputs)

        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hparams.num_units)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell,encoder_emb_inputs,time_major=True,dtype=tf.float32)

        #decoder
        self.decoder_inputs = tf.placeholder(tf.int32,shape=(self.hparams.decoder_length,self.hparams.batch_size),name="decoder_inputs")
        self.decoder_lengths = tf.placeholder(tf.int32,shape=(self.hparams.batch_size),name="decoder_length")

        self.embedding_decoder = tf.get_variable("embedding_decoder",[self.hparams.tgt_vocab_size,self.hparams.embedding_size])

        decoder_emb_inputs = tf.nn.embedding_lookup(self.embedding_decoder,self.decoder_inputs)

        #線形的に要素を引き伸ばす
        self.projection_layer = layers_core.Dense(self.hparams.tgt_vocab_size,use_bias = False)

        #helper
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs,self.decoder_lengths,time_major=True)

        self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hparams.num_units)

        #Attention付きかどうか判断して、それぞれ適した動作を行う
        if self.hparams.use_attention:
            #エンコーダーからの出力要素の順番を変更する
            attention_states = tf.transpose(encoder_outputs,[1,0,2])
            #LuongのAttentionを用いるのでその設定をする
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.hparams.num_units, attention_states,
                memory_sequence_length = None)
            #デコーダーとエンコーダーをくっつけたものを作る
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.decoder_cell,attention_mechanism,
                attention_layer_size = self.hparams.num_units)
            #デコーダーとエンコーダーの状態を結合する
            self.initial_state = self.decoder_cell.zero_state(self.hparams.batch_size,tf.float32).clone(cell_state=encoder_state)

        else:
            #エンコーダーの状態をそのまま初期状態として設定する
            self.initial_state = encoder_state

        #デコーダーの設定
        decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell,helper,self.initial_state,
            output_layer=self.projection_layer)

        #デコーダーの設定に基づいてdynamic_decoding
        final_outputs,_final_state,_final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
        #経過表示
        print("rnn_output.shape=", final_outputs.rnn_output.shape)
        print("sample_id.shape=", final_outputs.sample_id.shape)
        print("final_state=", _final_state)
        print("final_sequence_lengths.shape=", _final_sequence_lengths.shape)

        self.logits = final_outputs.rnn_output

        self.target_labels = tf.placeholder(tf.int32, shape=(self.hparams.batch_size, self.hparams.decoder_length))

        # Loss
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_labels, logits=self.logits)
        # Train
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.mask = tf.placeholder(tf.float32,shape=(self.hparams.batch_size,self.hparams.decoder_length),name="mask")
        self.loss = tf.multiply(self.loss,self.mask)

        # Calculate and clip gradients
        self.params = tf.trainable_variables()
        self.gradients = tf.gradients(self.loss, self.params)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.hparams.max_gradient_norm)

        # Optimization
        self.optimizer = tf.train.AdamOptimizer(self.hparams.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(self.clipped_gradients, self.params), global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def makeData(self,inIds,outIds):
        #inputのIDデータを読み込み
        ind_in = pd.read_csv(inIds,header = None)[0].values.tolist()
        #outputのIDデータを読み込み
        ind_out = pd.read_csv(outIds,header = None)[0].values.tolist()
        #input,outputそれぞれのデータを整形する(スペース区切りになっているものを要素ごとにリスト化する)
        #分割したデータを要素ごとに単語のみの配列に整形する
        self.in_ind = [ind_in[i].split(' ') for i in np.arange(len(ind_in))]
        self.in_ind = [np.delete(np.array(self.in_ind[i]),np.where(np.array(self.in_ind[i])=='')).tolist() for i in np.arange(len(self.in_ind))]
        self.out_ind = [ind_out[i].split(' ') for i in np.arange(len(ind_out))]
        self.out_ind = [np.delete(np.array(self.out_ind[i]),np.where(np.array(self.out_ind[i])=='')).tolist() for i in np.arange(len(self.out_ind))]
        #この時点でデータはそれぞれ[文章数][形態素数]の2次元配列になっている。
        #データの長さを調整
        #I/O　それぞれで一番長い文章の長さを取得する
        self.hparams.encoder_length = np.max([len(self.in_ind[i]) for i in np.arange(len(self.in_ind))]) + 1
        self.hparams.decoder_length = np.max([len(self.out_ind[i]) for i in np.arange(len(self.out_ind))]) + 1
        for i in np.arange(len(self.in_ind)):
            str_length = len(self.in_ind[i])
            for ind in np.arange(len(self.in_ind[i]),self.hparams.encoder_length):
                if len(self.in_ind[i])<self.hparams.encoder_length:
                    if str_length == ind:
                        self.in_ind[i].append(str(self.tgt_eos_id))
                    else:
                        self.in_ind[i].append(str(self.padding_id))

        for i in np.arange(len(self.out_ind)):
            str_length = len(self.out_ind[i])
            for ind in np.arange(len(self.out_ind[i]),self.hparams.decoder_length):
                if len(self.out_ind[i])<self.hparams.decoder_length:
                    if str_length == ind:
                        self.out_ind[i].append(str(self.tgt_eos_id))
                    else:
                        self.out_ind[i].append(str(self.padding_id))

        self.out_ind_sos=[]
        for i in np.arange(len(self.out_ind)):
            self.out_ind_sos.append([str(self.tgt_sos_id) if ind==0 else self.out_ind[i][ind-1] for ind in np.arange(len(self.out_ind[i]))])

        #マスクを作成する
        self.in_ind_mask = []
        for i in np.arange(len(self.in_ind)):
            x = np.ones(len(self.in_ind[i]))
            x[np.where(np.array(self.in_ind[i]) == str(self.padding_id))]=0
            self.in_ind_mask.append(list(np.fix(x)))

        self.out_ind_mask = []
        for i in np.arange(len(self.out_ind)):
            x = np.ones(len(self.out_ind[i]))
            x[np.where(np.array(self.out_ind[i]) == str(self.padding_id))]=0
            self.out_ind_mask.append(list(np.fix(x)))

        #ミニバッチ用のランダムインデックスを作成する
        self.randInd_in = np.random.permutation(len(self.in_ind))
        self.randInd_out = np.random.permutation(len(self.out_ind))

    def nextbatch(self):
        #makeDataにて作成したデータを切り分ける
        sInd = self.hparams.batch_size*self.batchcnt
        eInd = sInd + self.hparams.batch_size

        in_batch = np.transpose(np.array(self.in_ind)[self.randInd_in[sInd:eInd]]).astype(int)
        out_batch = np.array(self.out_ind)[self.randInd_out[sInd:eInd]].astype(int)
        out_sos_batch = np.transpose(np.array(self.out_ind_sos)[self.randInd_out[sInd:eInd]]).astype(int)
        out_mask_batch = np.array(self.out_ind_mask)[self.randInd_out[sInd:eInd]].astype(int)

        self.batchcnt += 1
        #batchとして取得する範囲がデータ範囲を超えた場合はカウントを初期化する
        if (self.hparams.batch_size*self.batchcnt)+self.batchcnt > len(self.in_ind):
            self.batchcnt = 0
        return in_batch,out_batch,out_sos_batch,out_mask_batch

    def data_train(self,train_number):
        #学習メソッド
        for i in range(train_number):
            encoder_data,decoder_label,decoder_data,mask = self.nextbatch()
            feed_dict = {
                self.encoder_inputs:encoder_data,
                self.target_labels:decoder_label,
                self.decoder_inputs:decoder_data,
                self.decoder_lengths:np.ones((self.hparams.batch_size),dtype=int)*self.hparams.decoder_length,
                self.mask:mask
            }
            _, loss_value,logits_value = self.sess.run([self.train_op,self.loss,self.logits],feed_dict=feed_dict)
            if i%10==0:
                print('iteration={} loss={}'.format(i,np.mean(loss_value)))

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder,
            tf.fill([self.hparams.batch_size],self.tgt_sos_id),
            self.tgt_eos_id)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
            self.decoder_cell,inference_helper,self.initial_state,
            output_layer=self.projection_layer)

        source_sequence_length = self.hparams.encoder_length
        maximum_iterations =tf.cast(tf.round(tf.reduce_max(source_sequence_length)*2),tf.int32)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,maximum_iterations=maximum_iterations)
        translations = outputs.sample_id

        feed_dict = {
            self.encoder_inputs:encoder_data,
        }
        replies = self.sess.run([translations],feed_dict=feed_dict)
        print(replies)
        pdb.set_trace()

if __name__ == "__main__":
    hparams=tf.contrib.training.HParams(
        batch_size = 50,
        encoder_length = 5,
        decoder_length = 5,
        num_units = 6,
        src_vocab_size = 14521,
        embedding_size = 8,
        tgt_vocab_size = 14524,
        learning_rate = 0.01,
        max_gradient_norm = 5.0,
        beam_width = 9,
        use_attention = True,
    )

    s2s=Seq2Seq_tf(hparams,14521,14522,14523)

    s2s.makeData('test_data_ids_in.txt','test_data_ids_in.txt')
    s2s.set_seq2seq()
    s2s.data_train(1000)
