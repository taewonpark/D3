#!/usr/bin/env bash

NKEY=64
CODE_SIZE=256

python sort_of_clevr_generator.py

# D3 w/o F
for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
	./run.sh D3_woF 4 256 4 ${CODE_SIZE} ${NKEY} ${seed}
done

# D3 w/ F
for seed in 0000 1111 2222 3333 4444 5555 6666 7777 8888 9999; do
	./run.sh D3_wF 4 256 4 ${CODE_SIZE} ${NKEY} ${seed}
done
