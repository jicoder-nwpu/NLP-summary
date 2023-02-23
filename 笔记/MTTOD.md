##### 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py -version 2.1 -run_type train -backbone ./model_path/ -model_dir ./output_dir -batch_size 4
```

##### 推理命令

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py -run_type predict -ckpt ./model_path/ckpt-epoch10/ -output predict.json -batch_size 8
```

