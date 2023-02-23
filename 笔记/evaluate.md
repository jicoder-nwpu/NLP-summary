#### Evaluate

##### sklearn

```bash
pip install -U scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#####  [standardized evaluation script](https://github.com/Tomiinek/MultiWOZ_Evaluation)

###### 使用

```python
pip install git+https://github.com/Tomiinek/MultiWOZ_Evaluation.git@master

from mwzeval.metrics import Evaluator
...

e = Evaluator(bleu=True, success=False, richness=False)
my_predictions = {}
for item in data:
    my_predictions[item.dialog_id] = model.predict(item)
    ...
    
results = e.evaluate(my_predictions)
print(f"Epoch {epoch} BLEU: {results}")
```

###### 输入格式

```json
{
    "xxx0000" : [
        {
            "response": "Your generated delexicalized response.",
            "state": {
                "restaurant" : {
                    "food" : "eatable"
                }, ...
            }, 
            "active_domains": ["restaurant"]
        }, ...
    ], ...
}

response – Your generated delexicalized response. You can use either the slot names with domain names, e.g. restaurant_food, or the domain adaptive delexicalization scheme, e.g. food.

state – Optional, the predicted dialog state. If not present (for example in the case of policy optimization models), the ground truth dialog state from MultiWOZ 2.2 is used during the Inform & Success computation. Slot names and values are normalized prior the usage.

active_domains – Optional, list of active domains for the corresponding turn. If not present, the active domains are estimated from changes in the dialog state during the Inform & Success rate computation. If your model predicts the domain for each turn, place them here. If you use domains in slot names, run the following command to extract the active domains from slot names automatically:
```

