#### 1. 预训练语言模型

##### 1. [Salesforce](https://huggingface.co/Salesforce)/[bart-large-xsum-samsum](https://huggingface.co/Salesforce/bart-large-xsum-samsum)

论文：《Controllable Abstractive Dialogue Summarization with Sketch Supervision》

github：[仓库](https://github.com/salesforce/ConvSumm/tree/master/CODS)

使用：

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Salesforce/bart-large-xsum-samsum")

model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/bart-large-xsum-samsum")

conv =  [
"Jason: whats up? Any plan for this weekend?", 
"John: I'm thinking of go watch a movie, but not decide which yet.", 
"Debbie: What? I thought that now all the theaters are closed due to the pandamic?", 
"John: Oh! That's right. Then no idea what to do."
]

from transformers import pipeline
# summarizer = pipeline("summarization", model="Salesforce/bart-large-xsum-samsum", device=0)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)
text = "<s> {}".format(" <s> ".join(conv))
summary = summarizer(text, min_length=10, max_length=100, num_beams=4)[0]["summary_text"]
```

#### 2. 模型使用

##### 1. pipeline

可以便捷的执行inference任务，调用模型生成结果。

#### 3. 论文

1. [《Controllable Abstractive Dialogue Summarization with Sketch Supervision》](https://aclanthology.org/2022.findings-acl.302/)2021 ACL

2. [《Dialogue Summaries as Dialogue States (DS2)》](https://aclanthology.org/2021.findings-acl.454/)2022 ACL
2. 《》