#### train & eval
> 模型train 和 eval的参数应该相同，因为要加载训练阶段模型参数。

> 在生成回复任务中，train和eval过程是完全不同的。
- - -
> 在translate阶段，word by word生成词时，每一步喂给decoder i 个词，且生成i个词表大小的概率分布，但仅取最后一个。
![image-20220425221033293](C:\Users\93683\AppData\Roaming\Typora\typora-user-images\image-20220425221033293.png)
- - -
#### python
> python中 True = 1, False = 0; tensor == tensor 返回同样大小的boolean tensor