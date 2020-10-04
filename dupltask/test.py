import os
import numpy as np
from time import time
import argparse
from dupltask.train import Trainer, get_sample, generate
from models import Reformer
from modules import Config


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch_size', default=100, type=int)
    ap.add_argument('-n', '--num_batch', default=100, type=int)
    ap.add_argument('-dir', '--model_dir', default='log_dir/lsh_seq32_nr1_bs4_manual', type=str)
    ap.add_argument('-tnr', '--test_num_hashes', default=4, type=int)
    ap.add_argument('-tbs', '--test_bucket_size', default=None, type=int)
    ap.add_argument('-uf', '--use_full', default=False, type=bool)
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
    ff_chunk_size = config['ff_chunk_size']
    seq_len = config['seq_len']
    seg_len = seq_len // 2 - 1

    batch_size = args.batch_size
    test_nr = args.test_num_hashes or num_hashes
    test_bs = args.test_bucket_size or bucket_size

    attn_config = Config({
        'dim': d_model,
        'num_heads': num_heads,
        'num_hashes': test_nr,
        'bucket_size': test_bs,
        'causality': True,
        'use_full': args.use_full
    })

    model = Reformer(d_model, d_ff, vocab_size, seq_len, num_blocks, attn_config, ff_chunk_size)

    log_dir = args.model_dir

    trainer = Trainer(model, checkpoint_dir=log_dir)
    trainer.load()

    avg_acc_arr = []
    # pre-build graph
    eval_data = np.stack([get_sample(vocab_size, seg_len) for _ in range(batch_size)])
    generate(model, seg_len, xs=eval_data)

    start_infer = time()
    for it in range(args.num_batch):
        tmp = time()
        eval_data = np.stack([get_sample(vocab_size, seg_len) for _ in range(batch_size)])
        gen_sample = generate(model, seg_len, xs=eval_data).numpy()

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