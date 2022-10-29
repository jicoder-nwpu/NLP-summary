##### 1. sqlite3 数据库引擎

[Andrea Madotto](https://dblp.uni-trier.de/pid/174/2905.html)



```python
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

model_name = "Salesforce/bart-large-xsum-samsum"
model_path = "./pretrained_model/"
cache_dir = "./pretrained_model/"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)


summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)
text = "<s> {}".format(" <s> ".join(conv))
summary = summarizer(text, min_length=10, max_length=100, num_beams=4)[0]["summary_text"]
print(summary)
```

##### MultiWOZ数据集组成

1. data.json

2. domain_db.json
3. dialogue_acts.json
4. ontology.json： 各个领域 slot 对应的 value
5. testListFile.json
6. valListFile.json
