[最新实验结果](./最新实验结果/)

| Summarization主实验                                          | Dataset |        |
| ------------------------------------------------------------ | ------- | ------ |
| Baseline                                                     | woz2.0  | woz2.1 |
| Transformer（12层）                                          | √       | √      |
| T5base                                                       | √       | √      |
| ResumTOD（上一篇文章结果）                                   | √       | √      |
| Summer（仅仅摘要模型）gen                                    | √       | √      |
| SummerTOD(联合回复生成)                                      | √       | √      |
| **Summarization消融实验（在Summer上实验）**                  |         |        |
| Model                                                        | WOZ2.0  | WOZ2.1 |
| 利用groundtruth摘要                                          | √       | √      |
| 对话历史选择3窗口                                            | √       | √      |
| dynamic filter选择6层（在全部对话历史作为输入的模型是做）    | √       | √      |
| 第一轮摘要使用用户说的话，而不是预定义句子                   | Titan   | √      |
| 输入去掉对话历史（只有上一轮的摘要和query）？？？            |         |        |
| **回复生成消融实验（在SummerTOD上实验）**                    |         |        |
| Model                                                        | WOZ2.0  | WOZ2.1 |
| 没有摘要模型（输入按照对话历史+状态+动作）                   | √       | √      |
| 没有共享Decoder，两个decoder                                 | Tesla   | √      |
| 回复的对话历史输入全部而非3窗口                              | Titan   | Titan  |
| Dynamic Fusion中对话历史编码和摘要编码的cross_attention交换先后顺序 |         |        |
| 生成的摘要不经过encoder，而直接使用隐状态（优先级最低）      |         |        |
| **其他分析实验**                                             |         |        |
| 实验类型                                                     | WOZ2.0  | WOZ2.1 |
| zero-shot迁移实验                                            |         |        |
| 单领域VS.多领域                                              |         |        |
| 多领域中随着对话长度增加（统计test集多领域中每个对话的轮数，根据轮数分别测试性能） |         |        |
| 对话历史窗口（只做回复生成时输入encoder的对话历史轮数）（1到6）优先级最高 | √       | √      |
| case study                                                   |         |        |
| 不同Prompt的影响（优先级最低）                               |         |        |