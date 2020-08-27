import os
import sys
import tensorflow as tf

from transformer_modules import positional_encoding
from modules import multihead_lsh_attention, ff, ReversibleBlock, ReversibleSequence


class TFModel:
    def save(self, sess, logdir, step):
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(logdir, model_name)
        sys.stdout.flush()
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.saver.save(sess, checkpoint_path, global_step=step)
        print("{} model has stored.".format(step))

    def load(self, sess, logdir):
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print("\tCheckpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sys.stdout.write('\t')
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            self.saver = tf.train.Saver(max_to_keep=5)
            return global_step
        else:
            print('No checkpoint found')
            return None


class Reformer(TFModel):
    def __init__(self, d_model, d_ff, num_heads, vocab_size, num_blocks, max_len, dropout_rate=0.0, is_training=True,
                 num_hashes=None, bucket_size=None, causality=False, is_full=False):
        assert num_hashes is not None
        assert bucket_size is not None

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.is_training = is_training
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.num_hashes = num_hashes
        self.bucket_size = bucket_size
        self.causality = causality
        self.is_full = is_full

        self.embeddings = tf.compat.v1.get_variable('weight_mat',
                                          dtype=tf.float32,
                                          shape=(vocab_size, d_model),
                                          initializer=tf.contrib.layers.xavier_initializer())

    def encode(self, xs, seq_len=None):
        # embedding
        enc = tf.nn.embedding_lookup(self.embeddings, xs)  # (N, T1, dim)
        enc *= self.d_model ** 0.5  # scale

        enc += positional_encoding(enc, self.max_len)
        enc = tf.layers.dropout(enc, self.dropout_rate, training=self.is_training)

        f = lambda x: multihead_lsh_attention(x, x, x,
                                              is_full=self.is_full,
                                              max_seq_len=self.max_len,
                                              seq_len=seq_len,
                                              num_hashses=self.num_hashes,
                                              bucket_size=self.bucket_size,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=self.is_training,
                                              causality=self.causality)
        g = lambda x: ff(x, num_units=[self.d_ff, self.d_model])

        ## Blocks
        blocks = [ReversibleBlock(f, g) for _ in range(self.num_blocks)]

        self.layer = ReversibleSequence(blocks)

        y1, y2 = self.layer(enc, enc)
        return enc, y1, y2

    def to_out(self, memory):
        weights = tf.transpose(self.embeddings)  # (d_model, vocab_size)
        result = tf.einsum('ntd,dk->ntk', memory, weights)  # (N, T, vocab_size)
        return result

    def create_loss(self, xs, labels, memory):
        raise NotImplementedError

    def train(self, xs, labels, lr, manual_grad=True):
        T = tf.constant(self.max_len, tf.int64)
        emb, y1, y2 = self.encode(xs, T)
        memory = tf.reduce_mean(tf.stack([y1, y2], 0), 0)

        # loss
        loss = self.create_loss(xs, labels, memory)

        if manual_grad:
            # TODO: Adam optimizer

            # grad and vars
            tf.compat.v1.get_variable_scope().reuse_variables()

            grads_list = []
            vars_list = []
            var_final = [tf.compat.v1.get_variable('weight_mat')]

            _grads = tf.gradients(loss, [y1, y2] + var_final, gate_gradients=True)
            dy1, dy2 = _grads[0], _grads[1]
            _grads = _grads[2:]

            grads_list.extend(_grads)
            vars_list.extend(var_final)

            with tf.control_dependencies(_grads):
                dy1, dy2 = tf.identity(dy1), tf.identity(dy2)

            y1, y2, dy1, dy2, _grads, _vars = self.layer.compute_gradients(y1, y2, dy1, dy2)
            grads_list.extend(_grads)
            vars_list.extend(_vars)

            emb_var = tf.compat.v1.get_variable('weight_mat')
            d_emb_dx1 = tf.gradients(emb, emb_var, dy1)
            d_emb_dx2 = tf.gradients(emb, emb_var, dy2)
            grads_list += d_emb_dx1
            grads_list += d_emb_dx2
            vars_list.extend([emb_var, emb_var])

            grad_and_vars = list(zip(tf.tuple(grads_list), vars_list))

            # for optimization
            opt = tf.train.GradientDescentOptimizer(lr)

        else:
            opt = tf.train.AdamOptimizer(lr)
            grad_and_vars = opt.compute_gradients(loss)

        train_op = opt.apply_gradients(grad_and_vars)
        self.saver = tf.train.Saver(max_to_keep=5)
        return loss, train_op, grad_and_vars