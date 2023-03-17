### 一、分析实验(2.1 & T5-base & Titan)

##### 1. encoder + (decoder x 2)

|             Input             | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :---------------------------: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|             ururu             |      2       | summary & response | 已完成 | 94.20  | 85.20  | 19.68 | 109.38 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_2_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_2_cross -batch_size 8 -context_size 3 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_2_cross.log & |
|             ururu             |      3       | summary & response | 已完成 | 94.20  | 85.90  | 19.80 | 109.85 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_3_cross.log & |
|             ururu             |      4       | summary & response | 已完成 | 94.50  | 85.90  | 19.57 | 109.77 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws4_cross_encoder | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws4_cross_encoder -batch_size 8 -context_size 5 -ururu -add_summary_cross_attention -warmup_ratio 0.1 & |
|             ururu             |      5       | summary & response | 已完成 | 95.30  | 86.50  | 18.45 | 109.35 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_5_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_5_cross -batch_size 8 -context_size 6 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_5_cross.log & |
|             ururu             |      6       | summary & response | 已完成 | 94.90  | 85.90  | 18.97 | 109.37 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_6_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_6_cross -batch_size 8 -context_size 7 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_6_cross.log & |
|             ururu             |      7       | summary & response | 已完成 | 94.50  | 85.90  | 19.37 | 109.57 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_7_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_7_cross -batch_size 8 -context_size 8 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_7_cross.log & |
|       ururu + truth_AC        |   best(3)    | summary & response | 已完成 | 93.40  | 89.60  | 29.92 | 121.42 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross | CUDA_VISIBLE_DEVICES=0 python3 main.py -run_type predict -ckpt sum_ws_3_cross/ckpt-epoch5 -output predict.json -batch_size 128 -use_true_curr_aspn -use_true_prev_aspn |
|             ubdar             |   best(3)    | summary & response | 已完成 | 94.40  | 85.70  | 19.38 | 109.43 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross_ubdar | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross_ubdar -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|          ururu(resp)          |     best     | summary & response | 已完成 | 93.40  | 84.80  | 19.61 | 108.71 |   6   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross_resp | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_resp -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|        ururu(no_cross)        |     best     | summary & response | 已完成 | 94.20  | 85.70  | 19.72 | 109.67 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_nocross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_nocross -batch_size 8 -context_size 4 -ururu -warmup_ratio 0.1 >> sum_ws_3_nocross.log & |
|   ururu(multi input fusion)   |   best(3)    | summary & response | 已完成 | 94.00  | 77.80  | 16.09 | 101.99 |  14   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_line_fusion | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_line_fusion -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_3_line_fusion.log & |
| ururu(自然语言替换分割占位符) |     best     | summary & response | 进行中 |        |        |       |        |       |                                                              |                                                              |

##### 2. encoder + decoder

| Input |  Output  |    Comment     | Status |
| :---: | :------: | :------------: | :----: |
| ururu | summary  | 摘要的评价指标 | 未开始 |
| ururu | response |       -        | 未开始 |

##### 3. cross_attention 可视化

##### 4. case study

### 二、分析实验(2.0 & T5-base & Titan)

##### 1. encoder + (decoder x 2)

|             Input             | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :---------------------------: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|             ururu             |      2       | summary & response | 已完成 | 94.20  | 84.80  | 20.43 | 109.93 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_2_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_2_cross -batch_size 8 -context_size 3 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_2_cross.log & |
|             ururu             |      3       | summary & response | 已完成 | 94.70  | 86.00  | 20.07 | 110.42 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_3_cross.log & |
|             ururu             |      4       | summary & response | 已完成 | 94.60  | 85.60  | 20.30 | 110.40 |   3   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_4_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_4_cross -batch_size 8 -context_size 5 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_4_cross.log & |
|             ururu             |      5       | summary & response | 已完成 | 94.60  | 85.40  | 20.06 | 110.06 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_5_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_5_cross -batch_size 8 -context_size 6 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_5_cross.log & |
|             ururu             |      6       | summary & response | 已完成 | 94.50  | 85.30  | 20.38 | 110.28 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_6_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_6_cross -batch_size 8 -context_size 7 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_6_cross.log & |
|             ururu             |      7       | summary & response | 已完成 | 94.10  | 84.90  | 20.05 | 109.55 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_7_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_7_cross -batch_size 8 -context_size 8 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_7_cross.log & |
|       ururu + truth_AC        |   best(3)    | summary & response | 已完成 | 94.40  | 90.00  | 30.11 | 122.31 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_3_cross.log & |
|             ubdar             |   best(3)    | summary & response | 已完成 | 93.90  | 84.50  | 20.40 | 109.60 |   6   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross_ubdar | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_ubdar -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|          ururu(resp)          |     best     | summary & response | 已完成 | 94.00  | 85.20  | 19.50 | 109.10 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross_resp | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_resp -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|        ururu(no_cross)        |     best     | summary & response | 已完成 | 93.70  | 86.10  | 19.88 | 109.78 |   3   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_nocross | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_nocross -batch_size 8 -context_size 4 -ururu -warmup_ratio 0.1 |
|   ururu(multi input fusion)   |     best     | summary & response | 已完成 | 92.90  | 82.10  | 17.60 | 105.10 |  19   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_line_fusion | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_line_fusion -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 -epochs 20 & |
| ururu(自然语言替换分割占位符) |     best     | summary & response | 进行中 |        |        |       |        |       |                                                              |                                                              |

##### 

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
