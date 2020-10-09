import os
import numpy as np
import tensorflow as tf
from time import time
import argparse
from models import Reformer, ReformerBlock
from modules import PositionalEncoder, Config

parent_dir = os.path.join(os.path.dirname(__file__), 'log_dir')
log_dir_tmpl = 'lsh_seq{}_nr{}_bs{}'
log_dir_full_attn_tmpl = 'lsh_seq{}_full'


class DuplTaskReformer(Reformer):
    def __init__(self, d_model, d_ff, vocab_size, max_len, num_blocks, attn_config,
                 ff_chunk_size=None, dropout_rate=0.0):
        super(Reformer, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.attn_config = attn_config

        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(max_len, masking=True, mask_val=0)

        self.blocks = []
        for i in range(num_blocks):
            reformer = ReformerBlock(d_model, d_ff, max_len, attn_config, ff_chunk_size, dropout_rate=dropout_rate)
            self.blocks.append(reformer)



def get_sample(vocab_size, seg_len):
    tmp = np.random.randint(1, vocab_size-1, size=seg_len)
    result = np.concatenate([[0], tmp, [0], tmp])
    return result


@tf.function
def loss_object(logits, labels):
    _, seq_len, _ = logits.shape
    seg_len = seq_len // 2 - 1
    logits = tf.slice(logits, [0, seq_len // 2, 0], [-1, seg_len, -1])
    probs = tf.nn.softmax(logits, -1)
    losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
    loss = tf.reduce_mean(losses)
    return loss, probs


@tf.function
def generate(model, seg_len, xs):
    start = time()
    result = xs[:, :seg_len + 2]  # 0w0

    for t in range(seg_len):
        _preds = model.ar_gen(result)
        _preds = tf.reshape(_preds, [-1, 1])
        result = tf.concat([result, _preds], axis=-1)

    print("AR generator has been built, elapsed:{0:.2}s".format(time()-start))
    return result


class Trainer:
    def __init__(
            self,
            model,
            loss_func=None,
            optimizer=None,
            checkpoint_dir='./checkpoints',
            batch_size=None,
            max_iter=50000,
    ):
        self.batch_size = batch_size
        self.model = model
        self.loss_object = loss_func
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.max_iter = max_iter

        self.print_every = 1000
        self.eval_every = 1000
        self.save_every = 10000

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

        # metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')

    def load(self):
        if self.checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        self.checkpoint.restore(
            self.checkpoint_manager.latest_checkpoint
        )

    def train(self, iterator, reset_checkpoint=False, manual_grad=True):
        train_log_dir = os.path.join(self.checkpoint_dir, 'logs')
        os.makedirs(train_log_dir, exist_ok=True)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        if not reset_checkpoint:
            self.load()
        else:
            print("reset and initializing from scratch.")

        start = time()

        for it in range(1, self.max_iter + 1):
            xs = iterator()
            y_true = xs[:, xs.shape[1] // 2 + 1:]

            loss, y_pred = self.model.train_step(xs, y_true, self.loss_object, self.optimizer, manual_grad=manual_grad)

            self.checkpoint.step.assign_add(1)

            self.train_loss(loss)
            self.train_accuracy(y_true, y_pred)

            if it % self.print_every == 0:
                end = time()
                msg = "Step: {0:}, Elapsed: {1:.3f}, Loss:{2:.3f}, Accuracy: {3:.3f}"
                print(msg.format(it, end - start, self.train_loss.result(), self.train_accuracy.result()))
                start = end

            if self.train_loss.result() < .5 and it % self.eval_every == 0:
                start_infer = time()
                seg_len = y_true.shape[-1]

                gen_sample_seed = y_true[0]
                gen_sample_seed = np.concatenate([[0], gen_sample_seed, [0]], -1).reshape(1, -1)

                gen_sample = generate(self.model, seg_len, xs=gen_sample_seed).numpy()
                print("Generated Sample:")
                print("\tleft:", gen_sample[0][:seg_len + 1])
                print("\tright:", gen_sample[0][seg_len + 1:])

                left_seg = gen_sample[:, 1:seg_len + 1]
                right_seg = gen_sample[:, seg_len + 2:]
                acc = np.sum(left_seg == right_seg, -1) / seg_len
                avg_acc = np.mean(acc)
                print("\taccuracy: {0:.4f}, elapsed: {1:.2f}s".format(avg_acc, time() - start_infer))

            if it % self.save_every == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step=it)
                    tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step=it)

                self.checkpoint_manager.save()
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch_size', default=8, type=int)
    ap.add_argument('-t', '--manual_grad', default=True, type=bool)
    ap.add_argument('-dm', '--d_model', default=64, type=int)
    ap.add_argument('-dff', '--d_ff', default=128, type=int)
    ap.add_argument('-nb', '--num_blocks', default=1, type=int)
    ap.add_argument('-nh', '--num_heads', default=4, type=int)
    ap.add_argument('-nr', '--num_hashes', default=1, type=int)
    ap.add_argument('-bs', '--bucket_size', default=8, type=int)
    ap.add_argument('-cs', '--ff_chunk_size', default=16, type=int)
    ap.add_argument('-l', '--seq_len', default=64, type=int)
    ap.add_argument('-vs', '--vocab_size', default=64, type=int)
    ap.add_argument('-lr', '--learning_rate', default=1e-3, type=float)
    ap.add_argument('-uf', '--use_full', default=False, type=bool)

    args = ap.parse_args()

    batch_size = args.batch_size
    d_model = args.d_model
    d_ff = args.d_ff

    num_blocks = args.num_blocks
    ff_chunk_size = args.ff_chunk_size

    vocab_size = args.vocab_size
    seq_len = args.seq_len
    learning_rate = args.learning_rate

    seg_len = seq_len // 2 - 1

    attn_config = Config({
        'dim': d_model,
        'num_heads': args.num_heads,
        'num_hashes': args.num_hashes,
        'bucket_size': args.bucket_size,
        'causality': True,
        'causal_start': None,
        'use_full': args.use_full
    })

    import os
    if attn_config.use_full:
        log_dir = log_dir_full_attn_tmpl.format(seq_len)
    else:
        log_dir = log_dir_tmpl.format(seq_len, attn_config.num_hashes, attn_config.bucket_size)

    if args.manual_grad:
        log_dir += "_manual"

    log_dir = os.path.join(parent_dir, log_dir)
    print("LOG DIR:", log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    import json
    argparse_dict = vars(args)
    with open(os.path.join(log_dir, 'config.json'), 'w') as fout:
        json.dump(argparse_dict, fout)

    model = DuplTaskReformer(d_model, d_ff, vocab_size, seq_len, num_blocks, attn_config, ff_chunk_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    trainer = Trainer(model, checkpoint_dir=log_dir, optimizer=optimizer, batch_size=batch_size,
                      max_iter=50000, loss_func=loss_object)

    def data_load():
        xs = np.stack([get_sample(vocab_size, seg_len) for _ in range(batch_size)])
        return xs

    trainer.train(iterator=data_load, reset_checkpoint=False, manual_grad=args.manual_grad)



