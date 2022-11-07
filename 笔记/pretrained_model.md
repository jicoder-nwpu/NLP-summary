#### 预训练模型使用

##### 模板框架

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

##### 学习率

```python
BERT --lr 2e-5 --lr_decay 0.95
```

