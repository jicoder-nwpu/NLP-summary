#### 预训练模型使用

##### multiwoz 模板框架

> data_process

> origin_model

> Model.py 自定义模型
>
> forward()  inference()

requirements.txt

run.py 多任务执行脚本

run.sh

> train.py
>
> class Mode train() validate() eval()

> utils.py
>
> _get_encoded_data() convert_batch()

##### 通用模板框架

###### data/ 放数据

###### dataset.py 数据集

```python
class InputExample
class InputFeature
def load_examples(file_path)     #读取文件 转为example
def convert_examples_to_features #将examples 转为feature 要符合模型输入格式
class _Dataset(torch.utils.data.Dataset):
    def __init__(self, features)
    def __len__(self)
    def __getitem__(self, idx)
DataLoader()
```

###### origin_model/ 放预训练参数

###### model.py 定义模型

1. train 初始化模型

从预训练参数中加载模型参数，使用from_pretrained()

> 不会加载通过register_buffer()注册的变量，但save()时会保存register_buffer()注册的变量

```python

```

2. test初始化模型

> 通过load_state_dict()加载保存的参数

```python

```

###### train.py 调用模型train，test

##### Bert 学习率

```python
BERT --lr 2e-5 --lr_decay 0.95
```

##### with **torch**.**no_grad**():

设置计算图不在传递梯度，不再占用显存

#### 对话系统融合知识

> content + knowledge

##### 表格

key - value

knowledge embedding = value embedding + key embedding + position embedding

##### 知识图谱

1. 通过template的形式转化为纯文本处理
2. 树结构

> soft position + masked encoder
>
> 同一课知识树中的词仅关注给树内的词

##### 文本

1. DOHA(Document Headed Attention)

共享编码器，知识太长需要截断

> context -> hc
>
> knowledge + context -> hd
>
> self-attention -> cross-attention -> corss-attention -> forward-feed

2. FID(Fusion in Decoder)

共享编码器，知识分段不需要截断

> 全部编码，连接后在Decoder中cross-attention
