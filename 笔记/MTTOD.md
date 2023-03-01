##### 训练命令

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py -version 2.1 -run_type train -backbone ./model_path/ -model_dir ./output_dir -batch_size 4

CUDA_VISIBLE_DEVICES=1 python3 main.py -version 2.1 -run_type train -backbone ./model_path/ -model_dir ./sum_ws_4_dir -context_size 5 -batch_size 8 -ururu -add_summary_cross_attention
```

##### 推理命令

```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py -run_type predict -ckpt ./model_path/ckpt-epoch10/ -output predict.json -batch_size 8
```



##### 仅用user + resp作为对话历史输入模型 -- 对话历史窗口大小2

###### MTTOD 仅bs与resp联合训练(不包含auxiliary_task)

![1677397415286](images/1677397415286.jpg)

###### OURS 仅summary与resp联合训练(不含cross_attention)

![image-20230226154355856](images/image-20230226154355856.png)

##### 用user + bs + db + ac + resp作为对话历史输入模型 -- 全部对话历史

###### MTTOD 仅bs与resp联合训练(不包含auxiliary_task)

![image-20230226154939860](images/image-20230226154939860.png)

##### 仅用user + resp作为对话历史输入模型 -- 对话历史窗口大小4

###### OURS 仅summary与resp联合训练(不含cross_attention)

![image-20230226155144299](images/image-20230226155144299.png)
