### 一、分析实验(2.1 & T5-base & Titan)

##### 1. encoder + (decoder x 2)

|           Input           | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :-----------------------: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|           ururu           |      2       | summary & response | 已完成 | 94.20  | 85.20  | 19.68 | 109.38 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_2_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_2_cross -batch_size 8 -context_size 3 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_2_cross.log & |
|           ururu           |      3       | summary & response | 已完成 | 94.20  | 85.90  | 19.80 | 109.85 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_3_cross.log & |
|           ururu           |      4       | summary & response | 已完成 | 94.50  | 85.90  | 19.57 | 109.77 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws4_cross_encoder | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws4_cross_encoder -batch_size 8 -context_size 5 -ururu -add_summary_cross_attention -warmup_ratio 0.1 & |
|           ururu           |      5       | summary & response | 已完成 | 95.30  | 86.50  | 18.45 | 109.35 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_5_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_5_cross -batch_size 8 -context_size 6 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_5_cross.log & |
|           ururu           |      6       | summary & response | 已完成 | 94.90  | 85.90  | 18.97 | 109.37 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_6_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_6_cross -batch_size 8 -context_size 7 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_6_cross.log & |
|           ururu           |      7       | summary & response | 已完成 | 94.50  | 85.90  | 19.37 | 109.57 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_7_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_7_cross -batch_size 8 -context_size 8 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_7_cross.log & |
|     ururu + truth_AC      |   best(3)    | summary & response | 已完成 | 93.40  | 89.60  | 29.92 | 121.42 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross | CUDA_VISIBLE_DEVICES=0 python3 main.py -run_type predict -ckpt sum_ws_3_cross/ckpt-epoch5 -output predict.json -batch_size 128 -use_true_curr_aspn -use_true_prev_aspn |
|           ubdar           |   best(3)    | summary & response | 已完成 | 94.40  | 85.70  | 19.38 | 109.43 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross_ubdar | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross_ubdar -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|        ururu(resp)        |   best(3)    | summary & response | 已完成 | 93.40  | 84.80  | 19.61 | 108.71 |   6   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross_resp | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_resp -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|      ururu(no_cross)      |   best(3)    | summary & response | 已完成 | 94.20  | 85.70  | 19.72 | 109.67 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_nocross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_nocross -batch_size 8 -context_size 4 -ururu -warmup_ratio 0.1 >> sum_ws_3_nocross.log & |
| ururu(multi input fusion) |   best(3)    | summary & response | 已完成 | 94.00  | 77.80  | 16.09 | 101.99 |  14   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_line_fusion | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_line_fusion -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> sum_ws_3_line_fusion.log |

##### 2. corss 热力图

##### 3. case study

### 二、分析实验(2.0 & T5-base & Titan)

##### 1. encoder + (decoder x 2)

|           Input           | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :-----------------------: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|           ururu           |      2       | summary & response | 已完成 | 94.20  | 84.80  | 20.43 | 109.93 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_2_cross | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_2_cross -batch_size 8 -context_size 3 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_2_cross.log & |
|           ururu           |      3       | summary & response | 已完成 | 94.70  | 86.00  | 20.07 | 110.42 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_3_cross.log & |
|           ururu           |      4       | summary & response | 已完成 | 94.60  | 85.60  | 20.30 | 110.40 |   3   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_4_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_4_cross -batch_size 8 -context_size 5 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_4_cross.log & |
|           ururu           |      5       | summary & response | 已完成 | 94.60  | 85.40  | 20.06 | 110.06 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_5_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_5_cross -batch_size 8 -context_size 6 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_5_cross.log & |
|           ururu           |      6       | summary & response | 已完成 | 94.50  | 85.30  | 20.38 | 110.28 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_6_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_6_cross -batch_size 8 -context_size 7 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_6_cross.log & |
|           ururu           |      7       | summary & response | 已完成 | 94.10  | 84.90  | 20.05 | 109.55 |   7   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_7_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_7_cross -batch_size 8 -context_size 8 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_7_cross.log & |
|     ururu + truth_AC      |   best(3)    | summary & response | 已完成 | 94.40  | 90.00  | 30.11 | 122.31 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 >> woz2_sum_ws_3_cross.log & |
|           ubdar           |   best(3)    | summary & response | 已完成 | 93.90  | 84.50  | 20.40 | 109.60 |   6   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross_ubdar | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_ubdar -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|        ururu(resp)        |   best(3)    | summary & response | 已完成 | 94.00  | 85.20  | 19.50 | 109.10 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross_resp | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_resp -batch_size 8 -context_size 4 -add_summary_cross_attention -warmup_ratio 0.1 & |
|      ururu(no_cross)      |   best(3)    | summary & response | 已完成 | 93.70  | 86.10  | 19.88 | 109.78 |   3   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_nocross | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_nocross -batch_size 8 -context_size 4 -ururu -warmup_ratio 0.1 |
| ururu(multi input fusion) |   best(3)    | summary & response | 已完成 | 92.90  | 82.10  | 17.60 | 105.10 |  19   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_line_fusion | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_line_fusion -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -warmup_ratio 0.1 -epochs 20 & |

##### 2. corss 热力图

##### 3. case study

### 二、MTTOD复现

|    Version    | Windows Size | Status | Inform | Sucess | Bleu  | Score  | Epoch |                         dir                         |
| :-----------: | :----------: | :----: | :----: | :----: | :---: | :----: | :---: | :-------------------------------------------------: |
|      2.0      |      3       | 已完成 | 93.30  | 83.90  | 19.14 | 107.74 |   9   | Titan/home/jhr/MTTOD-main/woz2_ws_3_add/ckpt-epoch9 |
|      2.1      |      3       | 已完成 | 91.70  | 83.40  | 20.14 | 107.69 |   8   |   Titan/home/jhr/MTTOD-main/ws_3_add/ckpt-epoch8    |
| 2.0(truth_bs) |      3       | 已完成 | 94.30  | 85.00  | 19.86 | 109.51 |   9   | Titan/home/jhr/MTTOD-main/woz2_ws_3_add/ckpt-epoch9 |
| 2.1(truth_bs) |      3       | 已完成 | 92.70  | 83.90  | 20.18 | 108.48 |   8   |   Titan/home/jhr/MTTOD-main/ws_3_add/ckpt-epoch8    |



### 三、分析实验(单/多领域 & T5-base & Titan)

##### 1. OURS

| Version |      Type       | Input | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :-----: | :-------------: | :---: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|   2.0   | single - single | ururu |      3       | summary & response | 已完成 | 90.13  | 82.06  | 15.17 | 101.27 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_3_cross_single | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_single -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -woz_type single -warmup_ratio 0.1 & |
|   2.0   |  multi - multi  | ururu |      3       | summary & response | 已完成 | 92.92  | 83.14  | 19.62 | 107.65 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_3_cross_multi | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_multi -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -woz_type multi -warmup_ratio 0.1 & |
|   2.0   | cross - single  | ururu |      3       | summary & response | 已完成 | 95.07  | 88.34  | 17.80 | 109.50 |   4   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_3_cross | python3 main.py -run_type predict -ckpt woz2_sum_ws_3_cross/ckpt-epoch4 -output predict.json -batch_size 32 |
|   2.0   |  cross - multi  | ururu |      3       | summary & response | 已完成 | 94.59  | 85.33  | 20.42 | 110.39 |   4   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_3_cross | python3 main.py -run_type predict -ckpt woz2_sum_ws_3_cross/ckpt-epoch4 -output predict.json -batch_size 32 |
|   2.1   | single - single | ururu |      3       | summary & response | 已完成 | 93.72  | 87.44  | 14.40 | 104.98 |   6   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_3_cross_single | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross_single -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -woz_type single -warmup_ratio 0.1 & |
|   2.1   |  multi - multi  | ururu |      3       | summary & response | 已完成 | 93.05  | 82.63  | 19.26 | 107.10 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_3_cross_multi | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross_multi -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -woz_type multi -warmup_ratio 0.1 & |
|   2.1   | cross - single  | ururu |      3       | summary & response | 已完成 | 93.72  | 88.34  | 17.45 | 108.48 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_3_cross | python3 main.py -run_type predict -ckpt sum_ws_3_cross/ckpt-epoch5 -output predict.json -batch_size 32 |
|   2.1   |  cross - multi  | ururu |      3       | summary & response | 已完成 | 94.34  | 85.20  | 20.16 | 109.93 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_3_cross | python3 main.py -run_type predict -ckpt sum_ws_3_cross/ckpt-epoch5 -output predict.json -batch_size 32 |

##### 2. MTTOD

| Version |      Type       | Windows Size | Status | Inform | Sucess | Bleu  | Score  | Epoch | Dir                                        | Command                                                      |
| :-----: | :-------------: | :----------: | :----: | :----: | :----: | :---: | :----: | :---: | ------------------------------------------ | ------------------------------------------------------------ |
|   2.0   | single - single |      3       | 已完成 | 95.52  | 85.20  | 14.78 | 105.14 |   9   | Titan/home/jhr/MTTOD-main/woz2_single_ws_3 | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_single_ws_3 -batch_size 8 -context_size 4 -add_auxiliary_task -warmup_ratio 0.1 -domain single >> woz2_single_ws_3.log |
|   2.0   |  multi - multi  |      3       | 已完成 | 90.86  | 81.47  | 19.35 | 105.52 |  10   | Titan/home/jhr/MTTOD-main/woz2_multi_ws_3  | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_multi_ws_3 -batch_size 8 -context_size 4 -add_auxiliary_task -warmup_ratio 0.1 -domain multi >> woz2_multi_ws_3.log |
|   2.0   | cross - single  |      3       | 已完成 | 92.38  | 85.65  | 16.74 | 105.75 |   9   | Titan/home/jhr/MTTOD-main/woz2_ws_3_add    | python3 main.py -run_type predict -ckpt woz2_ws_3_add/ckpt-epoch9 -output predict.json -batch_size 128 -domain single |
|   2.0   |  cross - multi  |      3       | 已完成 | 93.56  | 83.40  | 19.52 | 108.00 |   9   | Titan/home/jhr/MTTOD-main/woz2_ws_3_add    | python3 main.py -run_type predict -ckpt woz2_ws_3_add/ckpt-epoch9 -output predict.json -batch_size 128 -domain multi |
|   2.1   | single - single |      3       | 已完成 | 91.48  | 82.96  | 15.25 | 102.47 |  10   | Titan/home/jhr/MTTOD-main/single_ws_3      | python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./single_ws_3 -batch_size 8 -context_size 4 -add_auxiliary_task -warmup_ratio 0.1 -domain single >> single_ws_3.log |
|   2.1   |  multi - multi  |      3       | 已完成 | 91.12  | 81.85  | 19.70 | 106.19 |  10   | Titan/home/jhr/MTTOD-main/multi_ws_3       | python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./multi_ws_3 -batch_size 8 -context_size 4 -add_auxiliary_task -warmup_ratio 0.1 -domain multi >> multi_ws_3.log |
|   2.1   | cross - single  |      3       | 已完成 | 91.03  | 85.20  | 18.45 | 106.56 |   8   | Titan/home/jhr/MTTOD-main/ws_3_add         | python3 main.py -run_type predict -ckpt ws_3_add/ckpt-epoch8 -output predict.json -batch_size 128 -domain single |
|   2.1   |  cross - multi  |      3       | 已完成 | 91.89  | 82.88  | 20.41 | 107.79 |   8   | Titan/home/jhr/MTTOD-main/ws_3_add         | python3 main.py -run_type predict -ckpt ws_3_add/ckpt-epoch8 -output predict.json -batch_size 128 -domain multi |

#### 多领域窗口实验 OURS

| Version |     Type      | Input | Windows Size |       Output       | Status | Inform | Sucess | Bleu  | Score  | Epoch |                             dir                              | Command                                                      |
| :-----: | :-----------: | :---: | :----------: | :----------------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|   2.0   | multi - multi | ururu |      3       | summary & response | 已完成 | 92.92  | 83.14  | 19.62 | 107.65 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_3_cross_multi | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_3_cross_multi -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -woz_type multi -warmup_ratio 0.1 & |
|   2.0   | multi - multi | ururu |      4       | summary & response | 已完成 | 93.05  | 83.01  | 19.77 | 107.80 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_4_multi | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_4_multi -batch_size 8 -context_size 5 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.0   | multi - multi | ururu |      5       | summary & response | 已完成 | 93.31  | 82.37  | 19.61 | 107.45 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_5_multi | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_5_multi -batch_size 8 -context_size 6 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.0   | multi - multi | ururu |      6       | summary & response | 已完成 | 93.05  | 80.95  | 19.54 | 106.54 |   8   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_6_multi | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_6_multi -batch_size 8 -context_size 7 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.0   | multi - multi | ururu |      7       | summary & response | 已完成 | 93.82  | 83.01  | 19.62 | 108.04 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_sum_ws_7_multi | python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_sum_ws_7_multi -batch_size 8 -context_size 8 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.1   | multi - multi | ururu |      3       | summary & response | 已完成 | 93.72  | 87.44  | 14.40 | 104.98 |   6   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_3_cross_multi | CUDA_VISIBLE_DEVICES=1 nohup python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_3_cross_single -batch_size 8 -context_size 4 -ururu -add_summary_cross_attention -woz_type single -warmup_ratio 0.1 & |
|   2.1   | multi - multi | ururu |      4       | summary & response | 已完成 | 93.05  | 82.88  | 19.32 | 107.29 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_4_multi | python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_4_multi -batch_size 8 -context_size 5 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.1   | multi - multi | ururu |      5       | summary & response | 已完成 | 92.02  | 81.60  | 19.21 | 106.02 |   9   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_5_multi | python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_5_multi -batch_size 8 -context_size 6 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.1   | multi - multi | ururu |      6       | summary & response | 已完成 | 92.54  | 81.85  | 19.31 | 106.50 |  10   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_6_multi | python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_6_multi -batch_size 8 -context_size 7 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |
|   2.1   | multi - multi | ururu |      7       | summary & response | 已完成 | 92.92  | 83.14  | 19.37 | 107.40 |   9   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/sum_ws_7_multi | python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./sum_ws_7_multi -batch_size 8 -context_size 8 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type multi |

### 四、分析实验(单任务 & T5-base & Titan)

##### encoder + decoder

###### `***`表示最好模型上的结果

##### 1. summary

| version | Windows Size | Input |       Output       | Status |                           rouge-1                            |                           rouge-2                            |                           rouge-l                            | epoch |                             dir                              |
| :-----: | :----------: | :---: | :----------------: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :----------------------------------------------------------: |
|   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.922283129152879, 'p': 0.9392023019818028, 'f': 0.9255667254931568} | {'r': 0.8772182467207631, 'p': 0.8968897022209281, 'f': 0.8805116946287996} | {'r': 0.9215677058957571, 'p': 0.9384848894896802, 'f': 0.9248563942249765} |   5   | Titan/home/jhr/share_encoder_cross_attention/EncDec/sum_ws_3 |
|   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9189833591462238, 'p': 0.9136842625054972, 'f': 0.9106830368142944} | {'r': 0.8653010212198636, 'p': 0.8642902611476211, 'f': 0.8577363270338426} | {'r': 0.9177845844852724, 'p': 0.9125237350102509, 'f': 0.9095112124692017} |   1   | Titan/home/jhr/share_encoder_cross_attention/EncDec/sum_ws_3 |
|   2.1   |      3       | ururu | summary & response | 已完成 | {'r': 0.9228753163224764, 'p': 0.9367583873747523, 'f': 0.9244179638478859} | {'r': 0.877886232718826, 'p': 0.8949004185435945, 'f': 0.8795942811734093}, | {'r': 0.922155262527637, 'p': 0.9360557112666748, 'f': 0.9237121370850593} |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross |
|   2.0   |      3       | ururu |      summary       | 已完成 | {'r': 0.9299818567423722, 'p': 0.9426625320968237, 'f': 0.9316544078367516} | {'r': 0.8879356533679731, 'p': 0.9018309486940458, 'f': 0.8889221284407192} | {'r': 0.9294390116161602, 'p': 0.9421220066774841, 'f': 0.931115969795204} |   4   | Titan/home/jhr/share_encoder_cross_attention/EncDec/woz2_sum_ws_3 |
|   2.0   |      3       | ururu |      summary       | 已完成 | {'r': 0.9314036268970548, 'p': 0.9279415061624507, 'f': 0.9244615489014817} | {'r': 0.8847915258145336, 'p': 0.8821349137090325, 'f': 0.8769431719912371} | {'r': 0.9306600306180594, 'p': 0.9271958302773182, 'f': 0.9237223039444723} |   1   | Titan/home/jhr/share_encoder_cross_attention/EncDec/woz2_sum_ws_3 |
|   2.0   |      3       | ururu | summary & response | 已完成 | {'r': 0.9303882641050367, 'p': 0.9414149283787433, 'f': 0.9311982946793105} | {'r': 0.8873625023639897, 'p': 0.9004761245757127, 'f': 0.8879801120143638} | {'r': 0.9296462295537458, 'p': 0.9406734607532606, 'f': 0.9304609718604597} |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross |

##### 2. response

| version | Windows Size | Input |       Output       | Status | Inform | Sucess | Bleu  | Score  | epoch |                             dir                              |
| :-----: | :----------: | :---: | :----------------: | :----: | :----: | :----: | :---: | ------ | :---: | :----------------------------------------------------------: |
|   2.1   |      3       | ururu |      response      | 已完成 | 94.20  | 84.10  | 19.80 | 108.95 |   7   | Titan/home/jhr/share_encoder_cross_attention/EncDec/resp_ws_3 |
|   2.1   |      3       | ururu |      response      | 已完成 | 93.00  | 85.30  | 18.88 | 108.03 |   4   | Titan/home/jhr/share_encoder_cross_attention/EncDec/resp_ws_3 |
|   2.1   |      3       | ururu | summary & response | 已完成 | 94.20  | 85.90  | 19.80 | 109.85 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross |
|   2.0   |      3       | ururu |      response      | 已完成 | 93.70  | 85.20  | 19.93 | 109.38 |   5   | Titan/home/jhr/share_encoder_cross_attention/EncDec/woz2_resp_ws_3 |
|   2.0   |      3       | ururu |      response      | 已完成 | 93.30  | 83.60  | 20.32 | 108.77 |   8   | Titan/home/jhr/share_encoder_cross_attention/EncDec/woz2_resp_ws_3 |
|   2.0   |      3       | ururu | summary & response | 已完成 | 94.70  | 86.00  | 20.07 | 110.42 |   4   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/woz2_sum_ws_3_cross |

### 五、Cross Domain

##### OURS

|  Target Domain  | Version | Windows Size | Status | Inform | Sucess | Bleu  | Score | epoch |                             dir                              |                           commond                            |
| :-------------: | :-----: | :----------: | :----: | :----: | :----: | :---: | :---: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   restaurant    |   2.1   |      3       | 已完成 | 65.90  | 53.32  | 12.80 | 72.41 |   2   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_restaurant |                              -                               |
|   restaurant    |   2.0   |      3       | 已完成 | 65.90  | 50.50  | 12.80 | 71.00 |   4   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_restaurant |                              -                               |
|   attraction    |   2.1   |      3       | 已完成 | 75.33  | 64.67  | 12.36 | 82.36 |   6   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_attraction |                              -                               |
|   attraction    |   2.0   |      3       | 已完成 | 82.00  | 70.67  | 13.28 | 89.61 |   6   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_attraction |                              -                               |
|     police      |   2.1   |      3       | 已完成 |  100   | 33.47  | 13.46 | 80.19 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_hospital_police |                              -                               |
|     police      |   2.0   |      3       | 已完成 |  100   | 30.61  | 12.61 | 77.91 |   4   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_hospital_police |                              -                               |
|    hospital     |   2.1   |      3       | 已完成 |  100   | 41.81  | 12.65 | 83.56 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_hospital_police |                              -                               |
|    hospital     |   2.0   |      3       | 已完成 |  100   | 42.51  | 13.29 | 84.54 |   2   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_hospital_police |                              -                               |
|      hotel      |   2.1   |      3       | 已完成 | 84.75  | 78.62  | 7.02  | 88.70 |   2   | Tiatan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_hotel |                              -                               |
|      hotel      |   2.0   |      3       | 已完成 | 80.82  | 67.77  | 8.91  | 83.20 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_hotel |                              -                               |
|      train      |   2.1   |      3       | 已完成 | 83.19  | 64.35  | 5.05  | 78.82 |   2   | Tiatan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_train |                              -                               |
|      train      |   2.0   |      3       | 已完成 | 82.03  | 40.00  | 4.38  | 65.39 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_train |                              -                               |
|      taxi       |   2.1   |      3       | 已完成 |  100   | 18.62  | 8.84  | 68.15 |   5   | Tiatan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_taxi |                              -                               |
|      taxi       |   2.0   |      3       | 已完成 |  100   | 19.54  | 8.02  | 67.79 |   3   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_taxi |                              -                               |
| hospital_police |   2.1   |      3       | 已完成 |  100   | 37.97  | 13.00 | 81.99 |   5   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/cross_hospital_police | python3 devide_by_domain.py -version 2.1 -target_domains hospital-police<br />python3 preprocess.py -version 2.1 -window_size 3<br />python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./cross_hospital_police -batch_size 8 -context_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type cross -no_validation |
| hospital_police |   2.0   |      3       | 已完成 |  100   | 36.47  | 13.25 | 81.49 |   2   | Titan/home/jhr/share_encoder_cross_attention/Multi_Singel/woz2_cross_hospital_police | python3 devide_by_domain.py -version 2.0 -target_domains hospital-police<br />python3 preprocess.py -version 2.0 -window_size 3<br />python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_cross_hospital_police -batch_size 8 -context_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention -woz_type cross -no_validation |

##### MTTOD

|  Target Domain  | Version | Windows Size | Status | Inform | Sucess | Bleu  | Score | epoch |                         dir                          |                           commond                            |
| :-------------: | :-----: | :----------: | :----: | :----: | :----: | :---: | :---: | :---: | :--------------------------------------------------: | :----------------------------------------------------------: |
|   restaurant    |   2.1   |      3       | 已完成 |  8.09  |  6.94  | 9.43  | 16.94 |   2   |      Titan/home/jhr/MTTOD-main/cross_restaurant      |                              -                               |
|   restaurant    |   2.0   |      3       | 已完成 |  8.09  |  7.25  | 9.57  | 17.24 |   1   |   Titan/home/jhr/MTTOD-main/woz2_cross_restaurant    |                              -                               |
|   attraction    |   2.1   |      3       | 已完成 | 20.00  | 14.67  | 10.86 | 28.20 |   1   |      Titan/home/jhr/MTTOD-main/cross_attraction      |                              -                               |
|   attraction    |   2.0   |      3       | 已完成 | 20.00  | 14.67  | 10.75 | 28.08 |   1   |   Titan/home/jhr/MTTOD-main/woz2_cross_attraction    |                              -                               |
|     police      |   2.1   |      3       | 已完成 |  100   | 35.10  | 12.61 | 80.16 |   1   |   Titan/home/jhr/MTTOD-main/cross_hospital_police    |                              -                               |
|     police      |   2.0   |      3       | 已完成 |  100   | 31.43  | 13.57 | 79.28 |   5   | Titan/home/jhr/MTTOD-main/woz2_cross_hospital_police |                              -                               |
|    hospital     |   2.1   |      3       | 已完成 |  100   | 47.04  | 13.79 | 87.31 |   1   |   Titan/home/jhr/MTTOD-main/cross_hospital_police    |                              -                               |
|    hospital     |   2.0   |      3       | 已完成 |  100   | 44.25  | 13.52 | 85.64 |   5   | Titan/home/jhr/MTTOD-main/woz2_cross_hospital_police |                              -                               |
|      hotel      |   2.1   |      3       | 已完成 | 19.18  | 18.71  | 8.17  | 27.12 |   1   |        Tiatan/home/jhr/MTTOD-main/cross_hotel        |                              -                               |
|      hotel      |   2.0   |      3       | 已完成 | 19.18  | 18.08  | 6.40  | 25.04 |   1   |      Titan/home/jhr/MTTOD-main/woz2_cross_hotel      |                              -                               |
|      train      |   2.1   |      3       | 已完成 | 78.55  | 54.49  | 4.31  | 70.83 |   4   |        Tiatan/home/jhr/MTTOD-main/cross_train        |                              -                               |
|      train      |   2.0   |      3       | 已完成 | 78.55  | 54.49  | 6.59  | 73.11 |   1   |      Titan/home/jhr/MTTOD-main/woz2_cross_train      |                              -                               |
|      taxi       |   2.1   |      3       | 已完成 |  100   | 19.08  | 7.25  | 66.79 |   5   |        Tiatan/home/jhr/MTTOD-main/cross_taxi         |                              -                               |
|      taxi       |   2.0   |      3       | 已完成 |  100   | 20.00  | 6.21  | 66.21 |   6   |      Titan/home/jhr/MTTOD-main/woz2_cross_taxi       |                              -                               |
| hospital_police |   2.1   |      3       | 已完成 |  100   | 41.54  | 13.42 | 84.19 |   1   |   Titan/home/jhr/MTTOD-main/cross_hospital_police    | rm -r data/MultiWOZ_2.1/cross-processed<br />python3 devide_by_domain.py -version 2.1 -target_domains restaurant<br />python3 preprocess.py -version 2.1<br />python3 main.py -version 2.1 -run_type train -backbone model_path/ -model_dir ./cross_restaurant -batch_size 8 -context_size 4 -warmup_ratio 0.1 -add_auxiliary_task -domain cross -no_validation |
| hospital_police |   2.0   |      3       | 已完成 |  100   | 38.35  | 13.55 | 82.72 |   5   | Titan/home/jhr/MTTOD-main/woz2_cross_hospital_police | rm -r data/MultiWOZ_2.0/cross-processed<br />python3 devide_by_domain.py -version 2.0 -target_domains restaurant<br />python3 preprocess.py -version 2.0<br />python3 main.py -version 2.0 -run_type train -backbone model_path/ -model_dir ./woz2_cross_restaurant -batch_size 8 -context_size 4 -warmup_ratio 0.1 -add_auxiliary_task -domain cross -no_validation |

### 六、损失权重比例

###### 查询shell

```shell
grep -n '*$' resp_loss*_cross/result.txt | sort
```

| resp_loss_coeff | Version | Windows Size | Status | Inform | Sucess | Bleu  | Score  | epoch |                             dir                              |
| :-------------: | :-----: | :----------: | :----: | :----: | :----: | :---: | :----: | :---: | :----------------------------------------------------------: |
|       1.0       |   2.1   |      3       | 已完成 | 94.50  | 86.00  | 19.44 | 109.69 |   7   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss1_cross |
|       1.2       |   2.1   |      3       | 已完成 | 94.00  | 85.20  | 19.80 | 109.40 |   5   |     /home/jhr/share_encoder_cross_attention/MTTOD-main/      |
|       1.4       |   2.1   |      3       | 已完成 | 94.30  | 85.70  | 19.79 | 109.79 |   5   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss14_cross |
|       1.6       |   2.1   |      3       | 已完成 | 94.70  | 86.30  | 19.81 | 110.31 |   7   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss16_cross |
|       1.7       |   2.1   |      3       | 已完成 | 94.30  | 85.70  | 19.57 | 109.57 |   4   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss17_cross |
|       1.8       |   2.1   |      3       | 已完成 | 94.10  | 86.00  | 19.87 | 109.92 |   5   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss18_cross |
|       1.9       |   2.1   |      3       | 已完成 | 94.70  | 85.70  | 19.59 | 109.79 |   6   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss19_cross |
|       2.0       |   2.1   |      3       | 已完成 | 94.20  | 85.90  | 19.80 | 109.85 |   5   | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross |
|       2.1       |   2.1   |      3       | 已完成 | 94.60  | 85.70  | 19.74 | 109.89 |   5   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss21_cross |
|       2.2       |   2.1   |      3       | 已完成 | 94.70  | 86.60  | 19.75 | 110.40 |   7   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss22_cross |
|       2.3       |   2.1   |      3       | 已完成 | 94.40  | 85.00  | 19.64 | 109.34 |   7   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss23_cross |
|       2.4       |   2.1   |      3       | 已完成 | 94.70  | 86.20  | 20.02 | 110.47 |   5   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss24_cross |
|       2.5       |   2.1   |      3       | 已完成 | 93.70  | 85.50  | 19.72 | 109.32 |   6   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss25_cross |
|       2.6       |   2.1   |      3       | 已完成 | 94.10  | 85.30  | 19.81 | 109.51 |   4   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss26_cross |
|       2.7       |   2.1   |      3       | 已完成 | 94.40  | 86.00  | 19.62 | 109.82 |   6   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss27_cross |
|       2.8       |   2.1   |      3       | 已完成 | 94.80  | 86.40  | 19.80 | 110.40 |   4   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss28_cross |
|       2.9       |   2.1   |      3       | 已完成 | 94.70  | 86.00  | 19.83 | 110.18 |   5   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss29_cross |
|       3.0       |   2.1   |      3       | 已完成 | 95.00  | 85.80  | 19.75 | 110.15 |   7   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss30_cross |
|       3.1       |   2.1   |      3       | 已完成 | 94.60  | 86.10  | 19.85 | 110.20 |   5   | /home/jhr/share_encoder_cross_attention/MTTOD-main/resp_loss31_cross |



### Tips

###### Titan cross_attention版本需要在model初始化中手动设置是否添加summary cross attention模块

```python
decoder_config.add_summary_cross_attention = True
decoder_config.summary_attention_layers = 6
-add_summary_cross_attention
```

###### 新下载的`config.json`需添加```json"add_summary_cross_attention"=false``` 以及 `"summary_attention_layers": 0`

###### context_size(dia_history) = window_size(preprocess_summary_labels) + 1

#### 实验设置

epoch=10, batch_size=8, warmup_ratio=0.1
