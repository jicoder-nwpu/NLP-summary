#### MultiWOZ数据集组成

1. data.json

2. domain_db.json
3. dialogue_acts.json
4. ontology.json： 各个领域 slot 对应的 value
5. testListFile.json
6. valListFile.json

#### [DAMD](https://github.com/thu-spmi/damd-multiwoz/)

##### damd multiwoz preprocess

###### 文件结构

> data_processs/data/multiz_woz/

`annotated_user_da_with_span_full.json.zip`

data.json

`mapping.pair`

dialogue_acts.json

testListFile.json

valListFile.json

>  data_processs/db/

*_db.json

`value_set.json`

ontology.json

###### 执行顺序

python data_analysis.py

python preprocess.py

#### 基于模板的Summary

###### 对于无summary的样例，使用本轮resp作为labels

##### 依赖 /fine_tuning_summary

> import ontology
>
> from utils.data_loader import get_slot_information, EXCLUDE_DOMAINS
>
> from utils.fix_label import fix_general_label_error
>
> from utils.fix_label import has_or_character
>
> from utils.state_sum_converter import get_converter
>
> import json
>
> all_slot_values.json #用于summary state 去词化

##### delex_summary_by_annotation

```python
def delex_summary_by_annotation(self, dial_turn):
        u = dial_turn['summary']
        span = dial_turn['span_info']
        for s in span:
            slot = s[1]
            if slot == 'open':
                continue
            if ontology.da_abbr_to_slot_name.get(slot):
                slot = ontology.da_abbr_to_slot_name[slot]
            u.replace(s[2], '[value_'+slot+']')
        u = u.split()
        u_delex = ' '.join([t for t in u if t is not ''])
        u_delex = u_delex.replace('[value_address] , [value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_name] [value_name]', '[value_name]')
        u_delex = u_delex.replace('[value_name]([value_phone] )', '[value_name] ( [value_phone] )')
        return u_delex
```

###### 需要ontology.py

```python
all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
db_domains = ['restaurant', 'hotel', 'attraction', 'train']

# original slot names in goals (including booking slots)
# requestable_slots_in_goals = {
#     "taxi": ["car type", "phone"],
#     "police": ["postcode", "address", "phone"],
#     "hospital": ["address", "phone", "postcode"],
#     "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
#     "attraction": ["entrance fee", "type", "address", "postcode", "phone", "area", "reference"],
#     "train": ["duration", "leaveat", "price", "arriveby", "id", "reference"],
#     "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
# }

# informable_slots_in_goals = {
#     "taxi": ["leaveat", "destination", "departure", "arriveby"],
#     "police": [],
#     "hospital": ["department"],
#     "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
#     "attraction": ["area", "type", "name"],
#     "train": ["destination", "day", "arriveby", "departure", "people", "leaveat"],
#     "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
# }

normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

requestable_slots = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}
all_reqslot = ["car", "address", "postcode", "phone", "internet",  "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]
# count: 17

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
all_infslot = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
                     "leave", "destination", "departure", "arrive", "department", "food", "time"]
# count: 17

all_slots = all_reqslot + ["stay", "day", "people", "name", "destination", "departure", "department"]
get_slot = {}
for s in all_slots:
    get_slot[s] = 1
# count: 24


# mapping slots in dialogue act to original goal slot names
da_abbr_to_slot_name = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

# slot merging: not used currently
# slot_name_to_value_token = {
#     'entrance fee': 'price',
#     'pricerange': 'price',
#     'arrive': 'time',
#     'leave': 'time',
#     'departure': 'name',
#     'destination': 'name',
#     'stay': 'count',
#     'people': 'count',
#     'stars': 'count',
# }
# dialog_act_dom = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital', 'general', 'booking']
dialog_acts = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    # 'booking': ['book', 'inform', 'nobook', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}
all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)
# print(all_acts)

dialog_act_params = {
    'inform': all_slots + ['choice', 'open'] ,
    'request': all_infslot+['choice', 'price'],
    'nooffer': all_slots + ['choice'],
    'recommend': all_reqslot + ['choice', 'open'],
    'select': all_slots +['choice'],
    # 'book': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'nobook': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'offerbook':all_slots + ['choice'],
    'offerbooked': all_slots + ['choice'],
    'reqmore': [],
    'welcome': [],
    'bye': [],
    'greet': [],
}

# dialog_acts = ['inform', 'request', 'nooffer', 'recommend', 'select', 'book', 'nobook', 'offerbook', 'offerbooked',
#                         'reqmore', 'welcome', 'bye', 'greet'] # thank
dialog_act_all_slots = all_slots + ['choice', 'open']
# act_span_vocab = ['['+i+']' for i in dialog_act_dom] + ['['+i+']' for i in dialog_acts] + all_slots

# value_token_in_resp = ['address', 'name', 'phone', 'postcode', 'area', 'food', 'pricerange', 'id',
#                                      'department', 'place', 'day', 'count', 'car']
# count: 12


# special slot tokens in belief span
# no need of this, just covert slot to [slot] e.g. pricerange -> [pricerange]
slot_name_to_slot_token = {}


# special slot tokens in responses
# not use at the momoent
slot_name_to_value_token = {
    # 'entrance fee': '[value_price]',
    # 'pricerange': '[value_price]',
    # 'arriveby': '[value_time]',
    # 'leaveat': '[value_time]',
    # 'departure': '[value_place]',
    # 'destination': '[value_place]',
    # 'stay': 'count',
    # 'people': 'count'
}

special_tokens = ['<pad>', '<go_r>', '<unk>', '<go_b>', '<go_a>',
                            '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<go_d>','<eos_d>'] # 0,1,2,3,4,5,6,7,8,9,10

eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>',
    'summary': '<eos_r>', 'summary_gen': '<eos_r>'}
```

##### get_summary_bstate

```python
	"""
    整合基于模板的对话摘要生成
    """
    def get_summary_bstate(self, bstate, get_domain=False):
        """Based on the mturk annotations we form multi-domain belief state"""
        domains = [u'taxi',u'restaurant',  u'hospital', u'hotel',u'attraction', u'train', u'police']
        summary_bstate = []
        summary_bvalue = []
        active_domain = []
        for domain in domains:
            domain_active = False

            booking = []
            # print(domain,len(bstate[domain]['book'].keys()))
            for slot in sorted(bstate[domain]['book'].keys()):
                if slot == 'booked':
                    if len(bstate[domain]['book']['booked'])!=0:
                        booking.append(1)
                        # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                    else:
                        booking.append(0)
                else:
                    if bstate[domain]['book'][slot] != "":
                        booking.append(1)
                        _, res = clean_slot_values(domain, slot, bstate[domain]['book'][slot].strip().lower())
                        summary_bvalue.append(["{}-book {}".format(domain, slot.strip().lower()), res]) #(["book", domain, slot, bstate[domain]['book'][slot]])
                    else:
                        booking.append(0)
            if domain == 'train':
                if 'people' not in bstate[domain]['book'].keys():
                    booking.append(0)
                if 'ticket' not in bstate[domain]['book'].keys():
                    booking.append(0)
            summary_bstate += booking

            for slot in bstate[domain]['semi']:
                slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
                if bstate[domain]['semi'][slot] == 'not mentioned':
                    slot_enc[0] = 1
                elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                    slot_enc[1] = 1
                    summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), "dontcare"]) #(["semi", domain, slot, "dontcare"])
                elif bstate[domain]['semi'][slot]:
                    _, res = clean_slot_values(domain, slot, bstate[domain]['semi'][slot].strip().lower())
                    summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()), res]) #(["semi", domain, slot, bstate[domain]['semi'][slot]])
                if slot_enc != [0, 0, 0]:
                    domain_active = True
                summary_bstate += slot_enc

            # quasi domain-tracker
            if domain_active:
                summary_bstate += [1]
                active_domain.append(domain)
            else:
                summary_bstate += [0]

        #print(len(summary_bstate))
        assert len(summary_bstate) == 94
        if get_domain:
            return active_domain
        else:
            return summary_bstate, summary_bvalue
```

##### sys轮对对话历史处理

```python
new_ontology = {}
# 获取 state_to_sum 转换器
converter = get_converter('mwz')
with open('./all_slot_vlaues.json', 'r') as f:
    all_slot_values = json.loads(f.read())
    
    
    				"""
                    整合基于模板的对话摘要生成
                    """
                    #当前轮对话状态
                    _, belief_value_summary = self.get_summary_bstate(dial_state)
                    single_turn['state'] = {}
                    single_turn['state']["slot_values"] = {s[0]:s[1] for s in belief_value_summary}
                    for ss, vv in single_turn['state']["slot_values"].items():
                        if ss not in new_ontology:
                            new_ontology[ss] = []
                        if vv not in new_ontology[ss]:
                            new_ontology[ss].append(vv)
                    
                    SLOTS = get_slot_information(new_ontology)
                    slot_values = fix_general_label_error(single_turn['state']["slot_values"], SLOTS)
                    slot_values = {k: v for k, v in slot_values.items() if v != "none"}
                    
                    #窗口外对话状态
                    if turn_num > (window_size + 1) * 2 and raw_dial['log'][turn_num - (window_size + 1) * 2]['metadata']:
                        _, belief_value_summary = self.get_summary_bstate(raw_dial['log'][turn_num - (window_size + 1) * 2]['metadata'])
                        pre_slot_values = {s[0]:s[1] for s in belief_value_summary}
                        
                        pre_slot_values = fix_general_label_error(pre_slot_values, SLOTS)
                        pre_slot_values = {k: v for k, v in pre_slot_values.items() if v != "none"}

                        for k in pre_slot_values:
                            if k in slot_values and slot_values[k] == pre_slot_values[k]:
                                slot_values.pop(k)
                    single_turn['state']["slot_values"] = slot_values
                    # all_slot_values.update(slot_values)
                    summary_text = converter.state_to_sum(slot_values)
                    single_turn['summary'] = summary_text.lower()
                    for s in slot_values:
                        slot_values[s] = all_slot_values[s]
                    summary_delex = converter.state_to_sum(slot_values)
                    single_turn['summary_delex'] = summary_delex.lower()
```

#### sqlite3 数据库引擎

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
