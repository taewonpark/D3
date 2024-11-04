# Experiment for WikiText-103 task

## Prepare data

``` {bash}
./getdata.sh
```



## Training

``` {bash}
cd src
```
- Linear Transformer with D3 w/o F
``` {bash}
bash ../example_scripts/run.sh train --work_dir linear_with_D3_woF --seed 1111 --n_decomposition_layer 4 --code_size 64 --n_keys 64 --module_type r --attn_type 1004
```
- Linear Transformer with D3 w/ F
``` {bash}
bash ../example_scripts/run.sh train --work_dir linear_with_D3_wF --seed 1111 --n_decomposition_layer 4 --code_size 64 --n_keys 64 --module_type rf --attn_type 1004
```


## Validation

``` {bash}
bash ../example_scripts/run.sh valid --work_dir ${work_dir}
```

## Test

``` {bash}
bash ../example_scripts/run.sh eval --work_dir ${work_dir}
```
