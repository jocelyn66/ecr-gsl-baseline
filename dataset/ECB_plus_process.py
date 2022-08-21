import json

root_path = "../data/ECB+/interim/"
output_path = "../data/ECB+/processed_ECB+/"

for name in ['Dev', 'Test', 'Train']:
    corpus_name = "ECB_{}_corpus.txt".format(name)
    event_name = "ECB_{}_Event_gold_mentions.json".format(name)
    entity_name = "ECB_{}_Entity_gold_mentions.json".format(name)
    corpus_path = root_path + corpus_name
    event_path = root_path + event_name
    entity_path = root_path + entity_name
    output_file_path = output_path + "ECB_{}_processed_data.json".format(name)
    with open(corpus_path, 'r') as f:
        corpus_lines = f.readlines()
    with open(event_path, 'r') as f:
        event_data = json.load(f)
    with open(entity_path, 'r') as f:
        entity_data = json.load(f)
    
    output_data = []
    tokens = []
    doc_id = ''
    sent_id = ''
    for i in range(1, len(corpus_lines)):
        line = corpus_lines[i]
        if line=='' or line=='\n':
            assert tokens != []
            assert doc_id != ''
            assert sent_id != ''
            output_data.append({'text':' '.join(tokens), 'tokens':tokens, 
                                'doc_id':doc_id, 'sent_id':sent_id, 'event_coref':[], 'entity_coref':[]})
            tokens = []
            doc_id = ''
            sent_id = ''
        else:
            split = line.split()
            temp_doc_id = split[0]
            temp_sent_id = split[1]
            temp_token = split[3]            
            if doc_id == '':
                doc_id = temp_doc_id
            else:
                assert temp_doc_id == doc_id
            if sent_id == '':
                sent_id = temp_sent_id
            else:
                assert temp_sent_id == sent_id
            tokens.append(temp_token)
    assert tokens != []
    assert doc_id != ''
    assert sent_id != ''
    output_data.append({'text':' '.join(tokens), 'tokens':tokens, 
                        'doc_id':doc_id, 'sent_id':sent_id, 'event_coref':[], 'entity_coref':[]})
    
    doc2idx = {}
    for i in range(len(output_data)):
        corpus = output_data[i]
        if corpus['doc_id'] in doc2idx.keys():
            doc2idx[corpus['doc_id']].append(i)
        else:
            doc2idx[corpus['doc_id']] = [i]

    for i in range(len(event_data)):
        event = event_data[i]
        idx_list = doc2idx[event['doc_id']]
        for j in idx_list:
            if output_data[j]['sent_id'] == str(event['sent_id']):
                corpus_idx = j
                break
        assert corpus_idx is not None
        if not 'event_coref' in output_data[corpus_idx].keys():
            output_data[corpus_idx]['event_coref'] = []        
        output_data[corpus_idx]['event_coref'].append({'coref_chain':event['coref_chain'], 
                        'tokens_number':event['tokens_number'], 'tokens_str':event['tokens_str']})

    for i in range(len(entity_data)):
        entity = entity_data[i]
        idx_list = doc2idx[entity['doc_id']]
        for j in idx_list:
            if output_data[j]['sent_id'] == str(entity['sent_id']):
                corpus_idx = j
                break
        assert corpus_idx is not None
        if not 'entity_coref' in output_data[corpus_idx].keys():
            output_data[corpus_idx]['entity_coref'] = []        
        output_data[corpus_idx]['entity_coref'].append({'coref_chain':entity['coref_chain'], 
                        'tokens_number':entity['tokens_number'], 'tokens_str':entity['tokens_str']})


    with open(output_file_path, 'w', encoding='utf-8') as f:
        for output in output_data:
            json.dump(output, f)
            f.write('\n')