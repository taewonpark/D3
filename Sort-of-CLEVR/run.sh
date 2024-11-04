#!/bin/bash

model=$1
iterations=$2
dim=$3
heads=$4
code_size=$5
n_keys=$6
seed=$7
extras=$8

ext=${extras//\-\-/\_}
ext=${ext// /}

name="VIT_"$model"_"$iterations"_"$dim"_"$heads"_"$n_keys"_"$code_size"_"$seed""$ext

echo Running on $HOSTNAME
echo Running version $name
echo Extra arguments: $extras

PYTHONUNBUFFERED=1 python main.py --epochs 100 --relation-type binary --model $model \
--transformer-dim $dim --n-heads $heads --n-keys $n_keys --seed $seed \
--iterations $iterations --name $name --code-size $code_size