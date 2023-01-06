### 实验总结

#### summary生成

|  id  | 是否开放域预训练 | 是否DST摘要finetuning |            输入形式            |      生成label形式       | 窗口大小 | DST标签覆盖范围 | 是否完成 | BLEU  | Loss  |
| :--: | :--------------: | :-------------------: | :----------------------------: | :----------------------: | :------: | :-------------: | :------: | :---: | :---: |
|  a   |        是        |          否           |               -                |            -             |    -     |        -        |    是    |   -   |   -   |
|  b   |        是        |          是           |            user+sys            |     去词化的summary      |    5     |       all       |    是    | 40.7  | 0.277 |
|  c   |        是        |          是           |            user+sys            |     去词化的summary      |    5     |    当前窗口     |    是    | 57.75 | 0.629 |
|  d   |        是        |          是           |            user+sys            | 去词化的summary+response |    5     |       all       |    是    | 72.69 | 0.601 |
|  e   |        是        |          是           |            user+sys            | 去词化的summary+response |    5     |    当前窗口     |    是    | 78.24 | 0.695 |
|  f   |        是        |          是           |       user+sys_delex</s>       |    未去词化的summary     |    5     |    当前窗口     |    是    | 81.34 | 0.205 |
|  g   |        是        |          是           |              user              |    未去词化的summary     |    5     |    当前窗口     |    是    | 82.66 | 0.109 |
|  h   |        是        |          是           |       user+sys_delex</s>       |    未去词化的summary     |    9     |    当前窗口     |    是    | 80.96 | 0.240 |
|  i   |        是        |          是           | user<eos_u>+sys_nodelex<eos_r> |    未去词化的summary     |    5     |    当前窗口     |   ing    |       |       |
|  j   |        是        |          是           | user<eos_u>+sys_nodelex<eos_r> |    未去词化的summary     |    9     |    当前窗口     |   ing    |       |       |

| 模型 |                             位置                             |
| :--: | :----------------------------------------------------------: |
|  f   | 双卡/~/fine_tuning_summary/experiments/2023-01-02-15-28-23_all_sd557_lr3e-05_bs8_sp5_dc0.8_cw5_model_origin_model_1.0 |
|  h   | 双卡/~/fine_tuning_summary/experiments/2023-01-02-15-46-43_all_sd557_lr3e-05_bs4_sp5_dc0.8_cw9_model_origin_model_1.0 |
|      |                                                              |

##### 注意：

```python
#labels = '<s> </s>'
enc['summary'] = self.vocab.tokenizer.encode("<s> " + target_sen + " </s>")
```

#### response生成

>  该部分对话历史窗口与所使用的summary模型窗口保持一致

##### 纯Bart基线

| 使用的summary模型id | 是否完成 | Inform | Sucess | BLEU  | Total  | Loss  |
| :-----------------: | :------: | :----: | :----: | :---: | :----: | :---: |
|         无          |    是    | 65.47  | 58.86  | 21.97 | 84.135 | 1.139 |

##### 融合summary信息(decoder_hidden_states )

###### 1. 平均池化加入encoder_output

| 使用的summary模型id | 是否完成 | Inform | Sucess | BLEU  | Total  |  Loss  |
| :-----------------: | :------: | :----: | :----: | :---: | :----: | :----: |
| a(该实验可能有bug)  |    是    | 75.78  | 69.77  | 20.01 | 92.785 | 1.200  |
|          e          |    是    |   -    |   -    |   -   |   -    | 13.452 |
|          f          |    否    |        |        |       |        |        |
|          g          |    否    |        |        |       |        |        |

* 平均池化加入encoder_output的方式不适用于finetunin过`去词化的summary`的隐状态
* 未去词化的summary`待实验`

###### 2. 与decoder_input cross_attention

|  使用的summary模型id  | dropout | 是否完成 | Inform | Sucess | BLEU  | Total  | Loss  | 存疑 |
| :-------------------: | :-----: | :------: | :----: | :----: | :---: | :----: | :---: | :--: |
|           a           |   0.1   |    是    | 52.05  | 35.34  | 16.28 | 59.975 | 1.247 |      |
|           c           |   0.1   |    是    | 60.16  | 52.85  | 20.41 | 76.915 | 1.178 |      |
|     f(resp </s>)      |   0.1   |    是    | 73.87  | 60.76  | 19.8  | 87.115 | 1.159 |      |
| f(resp_nodelex </s>)  |   0.1   |    是    | 75.68  | 68.87  | 20.04 | 92.315 | 1.309 |      |
| f(summary_max_len=32) |   0.1   |    否    | 43.74  |  3.10  | 0.16  | 23.58  | 1.147 |  1   |
|           g           |   0.1   |    是    | 59.56  | 33.93  | 17.60 | 64.345 | 1.154 |      |
|           h           |   0.1   |    否    | 53.05  | 32.13  | 17.46 | 60.05  | 1.142 |  1   |
|    h(50%few_show)     |   0.1   |    否    | 62.96  | 56.46  | 20.15 | 79.86  | 1.150 |  1   |
|    h(25%few_show)     |   0.1   |    否    | 59.96  | 52.75  | 18.92 | 75.28  | 1.213 |  1   |
|    h(75%few_show)     |   0.1   |    否    | 72.37  | 56.46  | 18.49 | 82.905 | 1.169 |  1   |
|                       |         |          |        |        |       |        |       |      |

* 使用未去词化的summary信息更加有效
* sys和user对话历史分别经过encoder，效果不好；encoder中user与sys的self_attention操作是必要的

### 其他模型结果

|  Model  | Belief State | System Act | Inform | Success | BLEU | Total |
| :-----: | :----------: | :--------: | :----: | :-----: | :--: | :---: |
|  ARDM   |    oracle    |     -      |  87.4  |  72.8   | 20.6 | 100.7 |
| SOLOIST |    oracle    |     -      |  89.6  |  79.3   | 18.0 | 102.5 |
| SOLOIST |  generated   |     -      |  85.5  |  72.9   | 16.5 | 95.7  |
|  UBAR   |  generated   | generated  |  95.4  |  80.7   | 17.0 | 105.1 |
|  ours   |  generated   |     -      |  77.2  |  68.2   | 19.5 | 92.2  |

### 其他可能有影响的点

1. 对话窗口不完全导致部分对话状态缺失
2. 学习率以及下降率调参
3. MultiWOZ_2.0 和 MultiWOZ_2.1 难度差异
4. summary的质量还能提高

5. 共享encoder以及summary+response联合训练
