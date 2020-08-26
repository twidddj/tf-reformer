![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)

# Reformer 
This repository provides an implementation of [Reformer : Efficient Transformer](https://openreview.net/pdf?id=rkgNKkHtvB) in Tensorflow 1.x.

## Features
- [x] Synthetic task: duplicate a sequence of symbols 
- [x] Manual gradients (for GradientDescentOptimizer)
- [ ] Manual gradients (for AdamOptimizer)
- [ ] Seq2Seq (Encoder & Decoder)
- [ ] Real-world tasks

## Duplication Task
#### 1. train
```
> python -m dupltask.train --batch_size 8 --seq_len 32 --mode 'auto' --vocab_size 64 --unit_size 128 --num_blocks 1 --num_heads 4 --num_hashes 1 --bucket_size 4 --ff_chunk_size 4 --learning_rate 1e-3
```

#### 2. test
```
python -m dupltask.test --batch_size 200 --num_batch 10 --model_dir [model_path] --test_num_hashes 4 --test_bucket_size 4
```

* When bucket_size == seq_len, It's on full_attention mode. 

#### 3. short-cuts for evaluation
##### 1. test using different number of hashes on the same model ( `--test_num_hashes` )
```
python -m dupltask.test --test_num_hashes 1 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr2_bs4_manual
```
> `accuracy: 0.8214, elapsed: 23.25s`
```
python -m dupltask.test --test_num_hashes 2 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr2_bs4_manual
```
> `accuracy: 0.9683, elapsed: 30.19s`
```
python -m dupltask.test --test_num_hashes 4 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr2_bs4_manual
```
> `accuracy: 0.9960, elapsed: 47.57s`
```
python -m dupltask.test --test_num_hashes 8 -b 100 -n 100 -dir dupltask/log_dir/lsh_seq32_nr2_bs4_manual
```
> `accuracy: 0.9998, elapsed: 78.07s`

##### 2. test seq_len: 1024
```
python -m dupltask.test --test_num_hashes 2 -b 1 -n 10 -dir dupltask/log_dir/lsh_seq1024_nr2_bs64
```
> `accuracy: 0.9816, elapsed: 104.36s`

* Full-attention mode
```
python -m dupltask.test --test_bucket_size 1024 -b 1 -n 10 -dir dupltask/log_dir/lsh_seq1024_nr2_bs64
```
> `accuracy: 1.0000, elapsed: 145.70s`

## Issues
* You can currently only use GradientDescentOptimizer to apply the manual gradient.
* It works, but it can be extremely time consuming. ( `--mode 'auto'` is recommended then. )
* If directory which ends with '_manual', it had been trained using the manual gradient. Otherwise, it had been trained using auto gradient of AdamOptimizer.

## Requirements
* Code is tested on TensorFlow version 1.15 for Python 3.x.

## References
- [https://github.com/lucidrains/reformer-pytorch](https://github.com/lucidrains/reformer-pytorch)
- [https://github.com/renmengye/revnet-public](https://github.com/renmengye/revnet-public)
- [https://github.com/cerebroai/reformers](https://github.com/cerebroai/reformers)
- [https://github.com/Kyubyong/transformer](https://github.com/Kyubyong/transformer)