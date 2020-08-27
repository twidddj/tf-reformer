import numpy as np
import tensorflow as tf
from time import time
import argparse
from models import Reformer


log_dir_tmpl = 'log_dir/lsh_seq{}_nr{}_bs{}'
log_dir_full_attn_tmpl = 'log_dir/lsh_seq{}_full'


class DuplTaskReformer(Reformer):
    def __init__(self, d_model, d_ff, num_heads, vocab_size, num_blocks, max_len, dropout_rate=0.0,
                 is_training=True, num_hashes=None, bucket_size=None, is_full=False):

        super().__init__(d_model, d_ff, num_heads, vocab_size, num_blocks, max_len, dropout_rate=dropout_rate,
                 is_training=is_training, num_hashes=num_hashes, bucket_size=bucket_size, causality=True, is_full=is_full)

    def create_loss(self, xs, labels, memory):
        logits = self.to_out(memory)
        T = tf.shape(xs)[1]

        logits_target = tf.slice(logits, [0, T // 2, 0], [-1, T // 2 - 1, -1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_target)
        loss = tf.reduce_mean(tf.reduce_sum(loss, -1))
        return loss

    def build_for_ar_gen(self, N):
        self.xs = tf.placeholder(tf.int32, shape=[N, None], name='ar_input')
        self.T = tf.placeholder(tf.int64, name='ar_input_length')

        tT = self.bucket_size * 2
        pad_num = (tT - (self.T % tT)) % tT

        xs = tf.pad(self.xs, [[0, 0], [0, pad_num]])

        _, y1, y2 = self.encode(xs, seq_len=self.T)
        memory = tf.reduce_mean(tf.stack([y1, y2], 0), 0)
        logits = self.to_out(memory)

        self.saver = tf.train.Saver(max_to_keep=5)

        self.gen = logits
        self.N = N

    def ar_gen(self, sess, seg_len, samples=None, sample_func=None):
        if samples is None:
            assert sample_func is not None
            samples = np.stack([sample_func(self.vocab_size, seg_len) for _ in range(self.N)])

        assert samples is not None

        result = samples.copy()
        result = result[:, :seg_len + 2] #  0w0
        sample_len = result.shape[-1]

        _logits = sess.run(self.gen, feed_dict={self.xs: result, self.T: sample_len})
        _preds = np.argmax(_logits[:, sample_len-1], -1)
        for t in range(seg_len):
            sample_len += 1
            result = np.concatenate([result, _preds.reshape(-1, 1)], -1)
            _logits = sess.run(self.gen, feed_dict={self.xs: result, self.T: sample_len})
            _preds = np.argmax(_logits[:, sample_len-1], -1)
        return result


def get_sample(vocab_size, seg_len):
    tmp = np.random.randint(1, vocab_size-1, size=seg_len)
    result = np.concatenate([[0], tmp, [0], tmp])
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch_size', default=8, type=int)
    ap.add_argument('-t', '--mode', default='manual', choices=['auto', 'manual'])
    ap.add_argument('-dm', '--d_model', default=64, type=int)
    ap.add_argument('-dff', '--d_ff', default=128, type=int)
    ap.add_argument('-nb', '--num_blocks', default=1, type=int)
    ap.add_argument('-nh', '--num_heads', default=4, type=int)
    ap.add_argument('-nr', '--num_hashes', default=2, type=int)
    ap.add_argument('-bs', '--bucket_size', default=4, type=int)
    ap.add_argument('-l', '--seq_len', default=32, type=int)
    ap.add_argument('-vs', '--vocab_size', default=64, type=int)
    ap.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
    ap.add_argument('-f', '--is_full', default=True, type=bool)

    args = ap.parse_args()

    N = args.batch_size
    d_model = args.d_model
    d_ff = args.d_ff
    num_heads = args.num_heads
    num_blocks = args.num_blocks
    vocab_size = args.vocab_size
    num_hashes = args.num_hashes
    bucket_size = args.bucket_size
    seq_len = args.seq_len
    learning_rate = args.learning_rate
    is_full = args.is_full

    seg_len = seq_len // 2 - 1

    manual_grad = args.mode is not 'auto'

    import os

    if manual_grad:
        log_dir_tmpl += "_manual"

    if is_full:
        log_dir = log_dir_full_attn_tmpl.format(seq_len)
    else:
        log_dir = log_dir_tmpl.format(seq_len, num_hashes, bucket_size)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    import json
    argparse_dict = vars(args)
    with open(os.path.join(log_dir, 'config.json'), 'w') as fout:
        json.dump(argparse_dict, fout)

    xs = tf.placeholder(tf.int32, shape=[N, seq_len])
    ys = tf.slice(xs, [0, tf.shape(xs)[1]//2 + 1], [-1, -1])
    lr = tf.placeholder(tf.float32)

    model = DuplTaskReformer(d_model, d_ff, num_heads, vocab_size, num_blocks, seq_len,
                             num_hashes=num_hashes, bucket_size=bucket_size, is_full=is_full)
    loss, train_op, gvs = model.train(xs, ys, lr, manual_grad=manual_grad)
    model.build_for_ar_gen(1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        prev_step = model.load(sess, log_dir) or 0

        max_iter = 150000
        losses = []
        start = time()

        print_every = 1000
        eval_every = 1000
        save_every = 10000

        cur_loss = np.inf
        for it in range(prev_step + 1, max_iter + 1):
            _xs = np.stack([get_sample(vocab_size, seg_len) for _ in range(N)])
            _ys, _loss, _ = sess.run([ys, loss, train_op], feed_dict={xs: _xs, lr: learning_rate})

            losses.append(_loss)

            if it % print_every == 0:
                end = time()
                cur_loss = np.mean(losses)
                print("step:{0:} \telapsed: {1:.2f}s \t loss: {2:.3f}".format(it, end - start, cur_loss))
                losses = []
                start = end

            if cur_loss < 5 and it % eval_every == 0:
                start_infer = time()
                gen_sample = model.ar_gen(sess, seg_len, sample_func=get_sample)
                print("left:", gen_sample[0][:seg_len + 1])
                print("right:", gen_sample[0][seg_len + 1:])

                left_seg = gen_sample[:, 1:seg_len + 1]
                right_seg = gen_sample[:, seg_len + 2:]
                acc = np.sum(left_seg == right_seg, -1) / seg_len
                avg_acc = np.mean(acc)
                print("accuracy: {0:.4f}, elapsed: {1:.2f}s".format(avg_acc, time() - start_infer))

            if it % save_every == 0:
                model.save(sess, log_dir, it)
