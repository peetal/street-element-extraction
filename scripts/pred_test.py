from string import punctuation
import re, os
from transformers import TFAutoModelForTokenClassification, PretrainedConfig, BertConfig, AutoTokenizer, DataCollatorForTokenClassification
from collections import defaultdict
import numpy as np 
import pandas as pd
from datasets import Dataset

def clean(s):
    res = re.sub(r'(\w)(\()(\w)', '\g<1> \g<2>\g<3>', s)
    res = re.sub(r'(\w)([),.:;]+)(\w)', '\g<1>\g<2> \g<3>', res)
    res = re.sub(r'(\w)(\.\()(\w)', '\g<1>. (\g<3>', res)
    res = re.sub(r'\s+', ' ', res)
    res = res.strip()
    return res

def test_tokenization(batch): 
    return tokenizer(batch['tokens'], is_split_into_words=True)

def test_wordid(batch): 
    tokenized_input = tokenizer(batch['tokens'], is_split_into_words=True)
    batch_wordID = []
    for idx, _ in enumerate(batch['input_ids']):
        batch_wordID.append(tokenized_input.word_ids(batch_index = idx))
    return batch_wordID

def masking(ids,labels, mask):
    
    # masking ids and labels using mask
    masked_ids = np.ma.masked_where(mask==0,ids)
    masked_ids = np.ma.compressed(masked_ids)

    masked_labels = np.ma.masked_where(mask==0,labels)
    masked_labels = np.ma.compressed(masked_labels)

    return masked_ids, masked_labels

def compress_tag(masked_labels, wordid): 
    tag_compressed = []
    for id in range(0, np.max([i for i in wordid if i is not None])+1):

        # idx associated with the same word (across potentially multiple tokens)
        id_idx = np.where(np.asarray(wordid) == id)[0]
        id_tag = np.asarray(masked_labels)[id_idx]

        # compress tag
        poi = 0
        street = 0
        for t in id_tag: 
            if t in ['B-POI','E-POI','S-POI','S-POI']:
                poi+= 1
            elif t in ['B-STR','E-STR','S-STR','S-STR']:
                street+= 1
        if poi == 0 and street == 0: 
            tag_compressed.append("O")
        elif poi > street: 
            tag_compressed.append('POI')
        elif street > poi: 
            tag_compressed.append('STR')
        elif street == poi: # street = poi != 0 
            tag_compressed.append('POI/STR')
    return tag_compressed 

def recon_compress_tag(tag_compressed): 

    poi_idx = np.where(np.array(tag_compressed) == "POI")[0]
    street_idx = np.where(np.array(tag_compressed) == "STR")[0]
    if len(poi_idx) > 1: 
        for i in range(poi_idx[0], poi_idx[-1]+1): 
            tag_compressed[i] = "POI"
    if len(street_idx) > 1: 
        for i in range(street_idx[0], street_idx[-1]+1):
            tag_compressed[i] = "STR" 
    
    return tag_compressed


if __name__ == "__main__":
    
    # load test df
    test_df = pd.read_csv('test.csv')
    test_df['raw_address'] = test_df['raw_address'].apply(lambda x: x.strip())
    test_df['tokens'] = test_df['raw_address'].apply(clean).str.split()
    test_df = test_df.drop(columns = ['raw_address'])
    #test_df = test_df.head(1000)
    
    model_ckpt = "indobenchmark/indobert-base-p2" # specify model id 
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt) 
    
    # prepare test tf set
    ds_test = Dataset.from_pandas(test_df)
    ds_test_encoded = ds_test.map(test_tokenization, batched = True)

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="tf", padding = 'longest')
    tf_test_dataset = ds_test_encoded.to_tf_dataset(
        columns= ['input_ids', 'attention_mask'],
        shuffle=False,
        batch_size=512,
        collate_fn=data_collator
    )
    
    # load model: 
    finetuned_bert2_dir = "/home/peetal/hulacon/street-element-extraction/finetuned_bert2/"
    # load back model 
    config = BertConfig.from_json_file(os.path.join(finetuned_bert2_dir,"config.json"))
    model =  TFAutoModelForTokenClassification.from_pretrained(os.path.join(finetuned_bert2_dir,"tf_model.h5"), config = config)
    
    pred_NE = defaultdict(list)
    for batch in tf_test_dataset: 
        # make prediction for this batch 
        pred_logits = model.predict(batch)['logits']
        pred_labels = np.argmax(pred_logits, axis=-1)

        # id2tag 
        pred_entity = []
        for seq in list(pred_labels): 
            entity = []
            for l in seq: 
                entity.append(list(model.config.id2label.values())[l])
            pred_entity.append(entity)
        batch['pred_labels'] = np.array(pred_entity)

        # to pandas
        batch_df= Dataset.from_dict(batch).to_pandas()

        # reconstruct street-related element for each row 
        for _ ,row in batch_df.iterrows():
            mask = row['attention_mask']
            ids = row['input_ids']
            labels = row['pred_labels']

            # remove padding
            masked_ids, masked_labels = masking(ids, labels, mask)
            # convert ids to tokens 
            masked_tokens = tokenizer.convert_ids_to_tokens(masked_ids)

            # get word id for reconstructing tags
            raw_string = tokenizer.decode(masked_ids)
            raw_words = raw_string.split(" ")[1:-1]
            wordid = tokenizer(raw_words,is_split_into_words=True).word_ids()

            # reconstruct tokens into words, and compress tags accordingly. 
            compressed_tag = compress_tag(masked_labels, wordid)
            compressed_tag = recon_compress_tag(compressed_tag)
            #print(compressed_tag)
            street, poi = [],[]
            for word, tag in zip(raw_words, compressed_tag): 
                if tag == "POI" or tag == "POI/STR":
                    poi.append(word)
                elif tag == "STR" or tag == "POI/STR": 
                    street.append(word)
            pred_NE['POI/street'].append(" ".join(poi) + "/" + " ".join(street))
            
            
    # output file 
    pred_NE['id'] = list(range(len(pred_NE['POI/street'])))
    output_df = pd.DataFrame(pred_NE)
    output_df = output_df[['id', 'POI/street']]
    output_df.to_csv(os.path.join(finetuned_bert2_dir,"pred.csv"), index = False)
    