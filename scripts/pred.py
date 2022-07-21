from string import punctuation
import re, os
import tqdm
from transformers import TFAutoModelForTokenClassification, PretrainedConfig, BertConfig, AutoTokenizer, DataCollatorForTokenClassification
from collections import defaultdict
import numpy as np 
import pandas as pd
from datasets import Dataset
import logging
import argparse
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR)


parser = argparse.ArgumentParser(description='Post-fmriprep pipeline for DIVATTEN')
parser.add_argument(
    '--address',
    required=True,
    action='store',
    help='raw address')
args = parser.parse_args()

class AddressElementExtract(): 

    def __init__(self): 
        
        finetuned_bert2_dir = "/Users/peetal/Desktop/street-element-extraction/finetuned_bert2"
        model_ckpt = "indobenchmark/indobert-base-p2" # specify model id 
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt) 
        self.config = BertConfig.from_json_file(os.path.join(finetuned_bert2_dir,"config.json"))
        self.model = TFAutoModelForTokenClassification.from_pretrained(os.path.join(finetuned_bert2_dir,"tf_model.h5"), config = self.config)

    # for data cleaning 
    def _clean(self, s):
        res = re.sub(r'(\w)(\()(\w)', '\g<1> \g<2>\g<3>', s)
        res = re.sub(r'(\w)([),.:;]+)(\w)', '\g<1>\g<2> \g<3>', res)
        res = re.sub(r'(\w)(\.\()(\w)', '\g<1>. (\g<3>', res)
        res = re.sub(r'\s+', ' ', res)
        res = res.strip()
        return res 

    # tokenize the words
    def _test_tokenization(self, batch): 
        return self.tokenizer(batch['tokens'], is_split_into_words=True)

    # get testw wrod id 
    def _test_wordid(self, batch): 
        tokenized_input = self.tokenizer(batch['tokens'], is_split_into_words=True)
        batch_wordID = []
        for idx, _ in enumerate(batch['input_ids']):
            batch_wordID.append(tokenized_input.word_ids(batch_index = idx))
        return batch_wordID

    # use attention mask to remove padding and special tokens
    def _masking(self, ids,labels, mask):
        
        # masking ids and labels using mask
        masked_ids = np.ma.masked_where(mask==0,ids)
        masked_ids = np.ma.compressed(masked_ids)

        masked_labels = np.ma.masked_where(mask==0,labels)
        masked_labels = np.ma.compressed(masked_labels)

        return masked_ids, masked_labels

    # align label and reconstruct tag
    def _compress_tag(self, masked_labels, wordid): 
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

    # reconstruct tag
    def _recon_compress_tag(self, tag_compressed): 

        poi_idx = np.where(np.array(tag_compressed) == "POI")[0]
        street_idx = np.where(np.array(tag_compressed) == "STR")[0]
        if len(poi_idx) > 1: 
            for i in range(poi_idx[0], poi_idx[-1]+1): 
                tag_compressed[i] = "POI"
        if len(street_idx) > 1: 
            for i in range(street_idx[0], street_idx[-1]+1):
                tag_compressed[i] = "STR" 
        return tag_compressed
    
    def extract_element(self, raw_text): 

        # load test df
        test_df = pd.DataFrame({"raw_address":[raw_text]})
        test_df['raw_address'] = test_df['raw_address'].apply(lambda x: x.strip())
        test_df['tokens'] = test_df['raw_address'].apply(self._clean).str.split()
        test_df = test_df.drop(columns = ['raw_address'])
        
              
        # prepare test tf set
        ds_test = Dataset.from_pandas(test_df)
        ds_test_encoded = ds_test.map(self._test_tokenization, batched = True)

        data_collator = DataCollatorForTokenClassification(self.tokenizer, return_tensors="tf", padding = 'longest')
        tf_test_dataset = ds_test_encoded.to_tf_dataset(
            columns= ['input_ids', 'attention_mask'],
            shuffle=False,
            batch_size=512,
            collate_fn=data_collator
        )
        
        for batch in tf_test_dataset: 
            # make prediction for this batch 
            pred_logits = self.model.predict(batch)['logits']
            pred_labels = np.argmax(pred_logits, axis=-1)

            # id2tag 
            pred_entity = []
            for seq in list(pred_labels): 
                entity = []
                for l in seq: 
                    entity.append(list(self.model.config.id2label.values())[l])
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
                masked_ids, masked_labels = self._masking(ids, labels, mask)
                # convert ids to tokens 
                masked_tokens = self.tokenizer.convert_ids_to_tokens(masked_ids)

                # get word id for reconstructing tags
                raw_string = self.tokenizer.decode(masked_ids)
                raw_words = raw_string.split(" ")[1:-1]
                wordid = self.tokenizer(raw_words,is_split_into_words=True).word_ids()

                # reconstruct tokens into words, and compress tags accordingly. 
                compressed_tag = self._compress_tag(masked_labels, wordid)
                compressed_tag = self._recon_compress_tag(compressed_tag)
                
                street, poi = [],[]
                for word, tag in zip(raw_words, compressed_tag): 
                    if tag == "POI" or tag == "POI/STR":
                        poi.append(word)
                    elif tag == "STR" or tag == "POI/STR": 
                        street.append(word)
        street = " ".join(street)
        poi = " ".join(poi)
        return f"Raw address: {raw_text} \nStreet: {street} \nPOI: {poi}"

if __name__ == "__main__":

    model = AddressElementExtract()
    elements = model.extract_element(args.address)
    print(elements)