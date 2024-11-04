#!/usr/bin/env bash

NKEY=64
TOPK=8
CODE_SIZE=32

# D3 w/o F
for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
	python train.py -log_dir=D3_woF/${seed} -ratio=0.0 -seed=${seed} -n_keys=${NKEY} -top_k=${TOPK} -code_size=${CODE_SIZE}
done

# D3 w/ F
for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
	python train.py -log_dir=D3_wF/${seed} -ratio=0.0 -seed=${seed} -n_keys=${NKEY} -top_k=${TOPK} -code_size=${CODE_SIZE} -filler True
done