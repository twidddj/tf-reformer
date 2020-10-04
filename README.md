# Reformer 
This repository provides an implementation of [Reformer : Efficient Transformer](https://openreview.net/pdf?id=rkgNKkHtvB) in Tensorflow.

## Features
- [x] Synthetic task: duplicate a sequence of symbols 
- [x] Manual gradients
- [ ] Seq2Seq (Encoder & Decoder)
- [ ] Real-world tasks

## Duplication Task
#### 1. train
```
> python -m dupltask.train --batch_size 8 --seq_len 32 --vocab_size 64 --d_model 64 --d_ff 128 --num_blocks 1 --num_heads 4 --num_hashes 1 --bucket_size 4 --ff_chunk_size 4 --learning_rate 1e-3
```

#### 2. test
```
python -m dupltask.test --batch_size 200 --num_batch 10 --model_dir [model_path] --test_num_hashes 4 --test_bucket_size 4
```

#### 3. short-cuts for evaluation
##### 1. test using different number of hashes on the same model ( `--test_num_hashes` )
```
python -m dupltask.test --test_num_hashes 1 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr1_bs4_manual
```
> `accuracy: 0.9097, elapsed: 14.53s`
```
python -m dupltask.test --test_num_hashes 2 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr1_bs4_manual
```
> `accuracy: 0.9858, elapsed: 18.78s`
```
python -m dupltask.test --test_num_hashes 4 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr1_bs4_manual
```
> `accuracy: 0.9964, elapsed: 30.07s`
```
python -m dupltask.test --use_full 1 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr1_bs4_manual
```
> `accuracy: 0.9986, elapsed: 13.34s`

##### 2. test seq_len: 1024
```
python -m dupltask.test --test_num_hashes 2 -b 1 -n 10 -dir dupltask/log_dir/lsh_seq1024_nr2_bs64
```
> `accuracy: 0.9710, elapsed: 85.33s`

* Full-attention mode ( set `--use_full 1`)
```
python -m dupltask.test --use_full 1 -b 1 -n 10 -dir dupltask/log_dir/lsh_seq1024_nr2_bs64
```
> `accuracy: 1.0000, elapsed: 106.72s`

## Requirements
* Code is tested on TensorFlow version 2.3.0 for Python 3.7.

## References
- [https://github.com/lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)
- [https://github.com/renmengye/revnet-public](https://github.com/renmengye/revnet-public)
- [https://github.com/cerebroai/reformers](https://github.com/cerebroai/reformers)
- [https://github.com/Kyubyong/transformer](https://github.com/Kyubyong/transformer)