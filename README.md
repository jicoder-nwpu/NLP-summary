### 一、分析实验(2.1 & T5-base & Titan)

##### 1. encoder + (decoder x 2)

|             Input             | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :---------------------------: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|             ururu             |      2       | summary & response | 已完成 | 94.20  | 85.20  | 19.68 | 109.38 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_2_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_2_cross -batch_size 8 -context_size 3 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_2_cross.log & |
|             ururu             |      3       | summary & response | 已完成 | 94.20  | 85.90  | 19.80 | 109.85 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_3_cross.log & |
|             ururu             |      4       | summary & response | 已完成 | 94.50  | 85.90  | 19.57 | 109.77 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws4_cross_encoder | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws4_cross_encoder -batch_size 8 -context_size 5 -ururu -add_summary_cross_attention -warmup_ratio 0.1 & |
|             ururu             |      5       | summary & response | 已完成 | 95.30  | 86.50  | 18.45 | 109.35 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_5_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_5_cross -batch_size 8 -context_size 6 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_5_cross.log & |
|             ururu             |      6       | summary & response | 进行中 |        |        |       |        |       | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_6_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_6_cross -batch_size 8 -context_size 7 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_6_cross.log & |
|             ururu             |      7       | summary & response | 未开始 |        |        |       |        |       |                                                              |                                                              |
|       ururu + truth_AC        |     best     | summary & response | 未开始 |        |        |       |        |       |                                                              |                                                              |
|             ubdar             |     best     | summary & response | 未开始 |        |        |       |        |       |                                                              |                                                              |
|          ururu(resp)          |     best     | summary & response | 未开始 |        |        |       |        |       |                                                              |                                                              |
|        ururu(no_cross)        |     best     | summary & response | 未开始 |        |        |       |        |       |                                                              |                                                              |
|   ururu(multi input fusion)   |     best     | summary & response | 进行中 |        |        |       |        |       | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_line_fusion | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_line_fusion -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_3_line_fusion.log & |
| ururu(自然语言替换分割占位符) |     best     | summary & response | 未开始 |        |        |       |        |       |                                                              |                                                              |

##### 2. encoder + decoder

| Input |  Output  |    Comment     | Status |
| :---: | :------: | :------------: | :----: |
| ururu | summary  | 摘要的评价指标 | 未开始 |
| ururu | response |       -        | 未开始 |

##### 3. cross_attention 可视化

##### 4. case study

### 二、分析实验(2.0 & T5-base & Titan)

### 三、分析实验(单领域 & T5-base & Titan)



##### Tips: Titan cross_attention版本需要在model初始化中手动设置是否添加summary cross attention模块

```python
decoder_config.add_summary_cross_attention = True
decoder_config.summary_attention_layers = 6
-add_summary_cross_attention
```

###### 新下载的`config.json`需添加```json"add_summary_cross_attention"=false``` 以及 `"summary_attention_layers": 0`

###### context_size(dia_history) = window_size(preprocess_summary_labels) + 1

#### 实验设置

epoch=10, batch_size=8, warmup_ratio=0.1
