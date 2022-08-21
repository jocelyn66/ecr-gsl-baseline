import json


def add_new_item(dict, item, idx):
    if not item in dict.keys():
        dict[item] = idx
        idx +=1
    return idx

root_path = "../data/ECB/processed_ECB+/"

doc_schema = {"None":0}
event_shema = {"None":0}
entity_shema = {"None":0}
doc_idx = 1
event_idx = 1
entity_idx = 1

for name in ['Test']:  # ['Train', 'Dev', 'Test]
    print(name)
    data_path = root_path + "ECB_{}_processed_data.json".format(name)
    with open(data_path, 'r') as f:
        lines = f.readlines()    
    
    for i in range(len(lines)):
        line = lines[i]
        data = json.loads(line)
        doc_idx = add_new_item(doc_schema, data['doc_id'], doc_idx)
        for event_coref in data['event_coref']:
            event_idx = add_new_item(event_shema, event_coref['coref_chain'], event_idx)
        for entity_coref in data['entity_coref']:
            entity_idx = add_new_item(entity_shema, entity_coref['coref_chain'], entity_idx)
        
output_path = root_path + "ECB_Test_schema.json".format(name)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump([doc_schema, event_shema, entity_shema], f)