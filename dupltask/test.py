import os
import numpy as np
from time import time
import argparse
from dupltask.train import DuplTaskReformer, get_sample
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print(tf.__version__)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch_size', default=100, type=int)
    ap.add_argument('-n', '--num_batch', default=100, type=int)
    ap.add_argument('-dir', '--model_dir', default='log_dir/lsh_seq32_nr2_bs4_manual', type=str)
    ap.add_argument('-tnr', '--test_num_hashes', default=2, type=int)
    ap.add_argument('-tbs', '--test_bucket_size', default=None, type=int)
    ap.add_argument('-f', '--is_full', default=False, type=bool)
    args = ap.parse_args()

    assert args.model_dir is not None
    assert os.path.exists(args.model_dir)

    import json
    with open(os.path.join(args.model_dir, 'config.json')) as fin:
        config = json.load(fin)

    d_model = config['d_model']
    d_ff = config['d_ff']
    num_heads = config['num_heads']
    num_blocks = config['num_blocks']
    vocab_size = config['vocab_size']
    num_hashes = config['num_hashes']
    bucket_size = config['bucket_size']
    seq_len = config['seq_len']
    seg_len = seq_len // 2 - 1

    N = args.batch_size
    test_nr = args.test_num_hashes or num_hashes
    test_bs = args.test_bucket_size or bucket_size

    model = DuplTaskReformer(d_model, d_ff, num_heads, vocab_size, num_blocks, seq_len, is_training=False,
                             num_hashes=test_nr, bucket_size=test_bs, is_full=args.is_full)
    model.build_ar_gen(N)

    log_dir = args.model_dir

    avg_acc_arr = []
    start_infer = time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load(sess, log_dir)
        for it in range(args.num_batch):
            tmp = time()
            eval_data = np.stack([get_sample(vocab_size, seg_len) for _ in range(N)])
            gen_sample = model.ar_gen(sess, seg_len, samples=eval_data)

            left_seg = gen_sample[:, 1:seg_len + 1]
            right_seg = gen_sample[:, seg_len + 2:]
            acc = np.sum(left_seg == right_seg, -1) / seg_len
            avg_acc = np.mean(acc)
            avg_acc_arr.append(avg_acc)
            if it == 0:
                elapsed = time() - tmp
                estimated = elapsed * args.num_batch
                print("elapsed per batch: {0:.2f}s, estimated time: {1:.2f}s".format(elapsed, estimated))

    print("left:", gen_sample[0][:seg_len + 1])
    print("right:", gen_sample[0][seg_len + 1:])
    print("accuracy: {0:.4f}, elapsed: {1:.2f}s".format(np.mean(avg_acc_arr), time() - start_infer))