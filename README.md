<table style="text-align:center">
<tr>
<th colspan="15">TOD 实验结果</th>
</tr>
<tr>
<th rowspan="2">模型</th>
<th rowspan="2">backbone</th>
<th rowspan="2">trick</th>
<th rowspan="2">Summary labels窗口大小</th>
<th colspan="3">Encoder</th>
<th colspan="1">Decoder(Predict)</th>
<th colspan="2">实验设置</th>
<th colspan="4">得分</th>
<th rowspan="2">目录</th>
</tr>
<tr>
<th>ururu</th>
<th>resp/redx</th>
<th>context_size</th>
<th>use_truth_db</th>
<th>显卡</th>
<th>epoch</th>
<th>Inform</th>
<th>Success</th>
<th>Bleu</th>
<th>Score</th>
</tr>
<tr>
<th rowspan="7">MTTOD</th>
<th rowspan="7">T5-base</th>
<th>-</th>
<th>-</th>
<th>false</th>
<th>redx</th>
<th>-1*</th>
<th>false*</th>
<th>2080Ti</th>
<th>10</th>
<th>91.10</th>
<th>82.70</th>
<th>18.54</th>
<th>105.44</th>
<th>四卡/home/jhr/MTTOD-main/model_path</th>
</tr>
<tr>
<th>-</th>
<th>-</th>
<th>false</th>
<th>redx</th>
<th>-1*</th>
<th>true*</th>
<th>2080Ti</th>
<th>10</th>
<th>92.70</th>
<th>84.40</th>
<th>18.65</th>
<th>107.20</th>
<th>四卡/home/jhr/MTTOD-main/model_path</th>
</tr>
<tr>
<th>-</th>
<th>-</th>
<th>true</th>
<th>redx</th>
<th>2</th>
<th>true*</th>
<th>2080Ti</th>
<th>10</th>
<th>52.60</th>
<th>47.30</th>
<th>18.61</th>
<th>68.56</th>
<th>四卡/home/jhr/MTTOD-main/ururur_output_dir</th>
</tr>
<tr>
<th>add_auxiliary_task</th>
<th>-</th>
<th>false</th>
<th>redx</th>
<th>-1</th>
<th>false</th>
<th>Titan</th>
<th>9</th>
<th>89.90</th>
<th>81.20</th>
<th>18.66</th>
<th>104.21</th>
<th>Titan/home/jhr/MTTOD-main/output_dir</th>
</tr>
<tr>
<th>-</th>
<th>-</th>
<th>true</th>
<th>redx</th>
<th>-1</th>
<th>false</th>
<th>2080Ti</th>
<th>10</th>
<th>91.90</th>
<th>83.50</th>
<th>17.66</th>
<th>105.36</th>
<th>四卡/home/jhr/MTTOD-main/ururur_all_dir</th>
</tr>
<tr>
<th>add_auxiliary_task</th>
<th>-</th>
<th>true</th>
<th>redx</th>
<th>-1</th>
<th>false#</th>
<th>2080Ti</th>
<th>10</th>
<th>91.10</th>
<th>80.70</th>
<th>17.64</th>
<th>103.54</th>
<th>四卡/home/jhr/MTTOD-main/ururur_all_add_dir</th>
</tr>
<tr>
<th>-</th>
<th>-</th>
<th>false</th>
<th>redx</th>
<th>-1</th>
<th>false</th>
<th>Titan</th>
<th>10</th>
<th>90.60</th>
<th>82.10</th>
<th>18.14</th>
<th>104.49</th>
<th>Titan/home/jhr/MTTOD-main/noadd_output_dir</th>
</tr>
<tr>
<th rowspan="13">OURS</th>
<th rowspan="13">T5-base</th>
<th>-</th>
<th>2</th>
<th>false</th>
<th>redx</th>
<th>4</th>
<th>true</th>
<th>2080Ti</th>
<th>7</th>
<th>91.50</th>
<th>81.30</th>
<th>19.69</th>
<th>106.09</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/output_dir</th>
</tr>
<tr>
<th>-</th>
<th>2</th>
<th>true</th>
<th>redx</th>
<th>4</th>
<th>true</th>
<th>2080Ti</th>
<th>7</th>
<th>94.10</th>
<th>84.50</th>
<th>19.56</th>
<th>108.86</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/ururu_output</th>
</tr>
<tr>
<th>-</th>
<th>2</th>
<th>true</th>
<th>redx</th>
<th>4</th>
<th>true</th>
<th>2080Ti</th>
<th>7</th>
<th>94.10</th>
<th>84.50</th>
<th>19.56</th>
<th>108.86</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/again_output_dir</th>
</tr>
<tr>
<th>-</th>
<th>2</th>
<th>true</th>
<th>redx</th>
<th>2</th>
<th>true</th>
<th>2080Ti</th>
<th>10</th>
<th>93.80</th>
<th>84.50</th>
<th>18.70</th>
<th>107.85</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/ururu_context_size_2_output</th>
</tr>
<tr>
<th>-</th>
<th>2</th>
<th>true</th>
<th>redx</th>
<th>5</th>
<th>true</th>
<th>2080Ti</th>
<th>6</th>
<th>92.80</th>
<th>83.80</th>
<th>19.13</th>
<th>107.43</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/ws5_output_dir</th>
</tr>
<tr style="background:#bae8e8">
<th>cross attention/12 layers/summary id embedding encoder</th>
<th>4</th>
<th>true</th>
<th>redx</th>
<th>5(4)</th>
<th>true</th>
<th>Titan</th>
<th>7</th>
<th style="background:#ff7e1a">94.50</th>
<th style="background:#ff7e1a">85.90</th>
<th>19.57</th>
<th>109.77</th>
<th>Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws4_cross_encoder</th>
</tr>
<tr>
<th>cross attention/12 layers/decoder hidden states ff relu</th>
<th>4</th>
<th>true</th>
<th>redx</th>
<th>5(4)</th>
<th>true</th>
<th>Titan</th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_4_cross_ff_relu</th>
</tr>
<tr>
<th>cross attention/12 layers/decoder hidden states ff no relu</th>
<th>4</th>
<th>true</th>
<th>redx</th>
<th>5(4)</th>
<th>true</th>
<th>Titan</th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_4_cross_ff_norelu</th>
</tr>
<tr style="background:#e3f6f5">
<th>-</th>
<th>4</th>
<th>true</th>
<th>redx</th>
<th>5(4)</th>
<th>true</th>
<th>Titan</th>
<th>7</th>
<th>93.90</th>
<th>85.20</th>
<th style="background:#ff7e1a">19.86</th>
<th>109.41</th>
<th>Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws4_nocross_dir</th>
</tr>
<tr>
<th>-</th>
<th>3</th>
<th>true</th>
<th>redx</th>
<th>4(3)</th>
<th>true</th>
<th>2080Ti</th>
<th>10</th>
<th>93.50</th>
<th>84.40</th>
<th>18.79</th>
<th>107.74</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/sum_ws_3_dir</th>
</tr>
<tr>
<th>-</th>
<th>2</th>
<th>true</th>
<th>redx</th>
<th>3(2)</th>
<th>true</th>
<th>2080Ti</th>
<th>10</th>
<th>92.70</th>
<th>83.80</th>
<th>18.98</th>
<th>107.23</th>
<th>四卡/home/jhr/share_encoder/MTTOD-main/sum_ws_2_dir</th>
</tr>
</table>






##### # 表示变量可选择范围内的最好结果

##### * 表示不确定

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
