#### 一、对话摘要改进验证实验

##### 1. last_turn_summary cross_attention updated_history

|                         description                          | cross_attention层 | version | Windows Size | Input |       Output       | Status |                           rouge-1                            |                           rouge-2                            |                           rouge-l                            |      epoch       |                             dir                              |                           commond                            |
| :----------------------------------------------------------: | :---------------: | :-----: | :----------: | :---: | :----------------: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                              T5                              |         -         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9189833591462238, 'p': 0.9136842625054972, 'f': 0.9106830368142944} | {'r': 0.8653010212198636, 'p': 0.8642902611476211, 'f': 0.8577363270338426} | {'r': 0.9177845844852724, 'p': 0.9125237350102509, 'f': 0.9095112124692017} |        1         | Titan/home/jhr/share_encoder_cross_attention/EncDec/sum_ws_3 |                              -                               |
|                           ResumTOD                           |         -         |   2.1   |      3       | ururu | summary & response | 已完成 | {'r': 0.9228753163224764, 'p': 0.9367583873747523, 'f': 0.9244179638478859} | {'r': 0.877886232718826, 'p': 0.8949004185435945, 'f': 0.8795942811734093}, | {'r': 0.922155262527637, 'p': 0.9360557112666748, 'f': 0.9237121370850593} |        5         | Titan/home/jhr/share_encoder_cross_attention/MTTOD-main/sum_ws_3_cross |                              -                               |
| ***ground_truth***<br />**last_turn_summary** cross_attention updated_history |         1         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.8432920583693696, 'p': 0.8886613334754864, 'f': 0.8530655220513299} | {'r': 0.7854346418389616, 'p': 0.8054078773226111, 'f': 0.786730228966895} | {'r': 0.8422963205427837, 'p': 0.8876303721721008, 'f': 0.8520662119364042} |        6         |      四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross1       | CUDA_VISIBLE_DEVICES=3 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross1 -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_ws3_cross1.nouhp & |
| ***ground_truth***<br />**last_turn_summary** cross_attention updated_history |         6         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.8840279299429995, 'p': 0.9170213438910466, 'f': 0.8941206990710282} | {'r': 0.8217834184496179, 'p': 0.8560156058308698, 'f': 0.8310198425268335} | {'r': 0.8827541242106589, 'p': 0.9157032850475202, 'f': 0.89283487805432} | 1(2及之后过拟合) |      四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross6       | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross6 -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_ws3_cross6.nouhp & |
| ***ground_truth***<br />**last_turn_summary** cross_attention updated_history |         9         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9211874600163883, 'p': 0.9359779444895265, 'f': 0.9233048924397754} | {'r': 0.8734222452864813, 'p': 0.8914716220049742, 'f': 0.8757422113592979} | {'r': 0.9204482464671542, 'p': 0.9352651870569518, 'f': 0.9225855694992697} |        6         |      四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross9       | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross9 -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_ws3_cross9.nouhp & |
| ***ground_truth***<br />**last_turn_summary** cross_attention updated_history |        12         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9290662786316665, 'p': 0.9359515723658951, 'f': 0.9275268009345935} | {'r': 0.8856745469424403, 'p': 0.894081551891404, 'f': 0.8836049047816162} | {'r': 0.9284060310248881, 'p': 0.935305778287971, 'f': 0.9268790658342543} |        10        |      四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross12      | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross12 -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_ws3_<br/>cross12.nouhp & |
| ***generate***<br />**last_turn_summary** cross_attention updated_history |         1         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9283672927694834, 'p': 0.9374753155816531, 'f': 0.9277591463535587} | {'r': 0.8845623628321384, 'p': 0.894561387688253, 'f': 0.8830646117915214} | {'r': 0.9276505584801108, 'p': 0.9367555872561452, 'f': 0.9270467070508044} |        10        |    四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross1_gen     | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross1_gen -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_<br/>ws3_cross1_gen.nouhp & |
| ***generate***<br />**last_turn_summary** cross_attention updated_history |         6         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.8343892070635252, 'p': 0.889369883212498, 'f': 0.8452126486886589} | {'r': 0.77665696053042, 'p': 0.7993749190602365, 'f': 0.7792259505602457} | {'r': 0.8331064456551358, 'p': 0.8878468674086984, 'f': 0.8438563487002831} |    3(7过拟合)    |    四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross6_gen     | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross6_gen -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_<br/>ws3_cross6_gen.nouhp & |
| ***generate***<br />**last_turn_summary** cross_attention updated_history |         9         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9165614508327716, 'p': 0.9370384871046112, 'f': 0.9212947798549218} | {'r': 0.8699652659977982, 'p': 0.8917951112921833, 'f': 0.8740121207014243} | {'r': 0.91571820201225, 'p': 0.9361923347746697, 'f': 0.9204580473752539} | 3(6及之后过拟合) |    四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross9_gen     | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross9_gen -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_<br/>ws3_cross9_gen.nouhp & |
| ***generate***<br />**last_turn_summary** cross_attention updated_history |        12         |   2.1   |      3       | ururu |      summary       | 已完成 | {'r': 0.9267018259424699, 'p': 0.9351449366206656, 'f': 0.925776009142429} | {'r': 0.8825403159932376, 'p': 0.8923897580309682, 'f': 0.8810137653801428} | {'r': 0.9259515181556273, 'p': 0.9344022859336952, 'f': 0.925036211956771} |        10        |    四卡/home/jhr/query-sum/MTTOD-main/sum_ws3_cross12_gen    | CUDA_VISIBLE_DEVICES=0 nohup python3 main.py -run_type train -backbone model_path/ -model_dir ./sum_ws3_cross12_gen -context_size 4 -grad_accum_steps 2 -batch_size 4 -ururu -warmup_ratio 0.1 -add_summary_cross_attention >> sum_<br/>ws3_cross12_gen.nouhp & |
| ***generate***<br />last_turn_summary cross_attention **updated_history** |         1         |   2.1   |      3       | ururu |      summary       | 进行中 |                                                              |                                                              |                                                              |        3         |    四卡/home/jhr/query-sum/MTTOD-main/his_ws3_cross1_gen     |                              -                               |
| ***generate***<br />last_turn_summary cross_attention **updated_history** |         6         |   2.1   |      3       | ururu |      summary       | 进行中 |                                                              |                                                              |                                                              |        3         |    四卡/home/jhr/query-sum/MTTOD-main/his_ws3_cross6_gen     |                              -                               |
| ***generate***<br />last_turn_summary cross_attention **updated_history** |        12         |   2.1   |      3       | ururu |      summary       | 进行中 |                                                              |                                                              |                                                              |        10        |    四卡/home/jhr/query-sum/MTTOD-main/his_ws3_cross12_gen    |                              -                               |
