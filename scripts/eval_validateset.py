from transformers import TFAutoModelForTokenClassification, PretrainedConfig, BertConfig, AutoTokenizer, DataCollatorForTokenClassification
from tensorflow import keras
from datasets import Dataset, load_metric
import pandas as pd 
import numpy as np
import tensorflow as tf
import ast, os, pickle

# sub-word tokenization using pre-trained autotokenizer
def tokenize_and_align_labels(batch): 

    tag2int = {'B-POI':0, 'B-STR':1, 'E-POI':2, 'E-STR':3, 'I-POI':4,
           'I-STR':5, 'S-POI':6, 'S-STR':7, 'O':8}
           
    tokenized_inputs = tokenizer(batch['tokens'], is_split_into_words=True)
    labels=[]
    for idx, label in enumerate(batch['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index = idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids: 
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tag2int[label[word_idx]])
            else: 
                label_ids.append(tag2int[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels

    return tokenized_inputs

def encode_dataset(ds):
    return ds.map(tokenize_and_align_labels, batched= True, remove_columns=['tags','tokens', 'index'])

def evaluate(model, dataset, ner_labels):
        all_predictions = []
        all_labels = []
        for batch in dataset:
            logits = model.predict(batch)["logits"]
            labels = batch["labels"]
            predictions = np.argmax(logits, axis=-1)
            for prediction, label in zip(predictions, labels):
                for predicted_idx, label_idx in zip(prediction, label):
                    if label_idx == -100:
                        continue
                    all_predictions.append(ner_labels[predicted_idx])
                    all_labels.append(ner_labels[label_idx])
        return metric.compute(predictions=[all_predictions], references=[all_labels])

if __name__ == "__main__":
    
    finetuned_bert2_dir = "/home/peetal/hulacon/street-element-extraction/finetuned_bert2/"
    # load back model 
    config = BertConfig.from_json_file(os.path.join(finetuned_bert2_dir,"config.json"))
    model =  TFAutoModelForTokenClassification.from_pretrained(os.path.join(finetuned_bert2_dir,"tf_model.h5"), config = config)

    # Check performance on validation set: perform the split again 
    df_converters = {'tokens': ast.literal_eval, 'labels': ast.literal_eval}
    df = pd.read_csv("train_df_pretokenization.csv", converters=df_converters) 

    model_ckpt = "indobenchmark/indobert-base-p2" # specify model id 
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt) 

    # tokenization and align lables to sub-words
    df = df.rename(columns={"labels": "tags"})    
    ds = Dataset.from_pandas(df)
    ds_encoded = encode_dataset(ds)

    # split into training and validating 
    test_size=0.15
    processed_dataset = ds_encoded.shuffle(seed=42).train_test_split(test_size=test_size)

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="tf", padding = 'longest')

    # create tf datasets as model inputs 
    tf_val_dataset = processed_dataset['test'].to_tf_dataset(
        columns= ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        shuffle=False,
        batch_size=512,
        collate_fn=data_collator
    )
    
    # evaluate
    metric = load_metric("seqeval")
    results = evaluate(model, tf_val_dataset, ner_labels=list(model.config.id2label.values()))
    
    with open(os.path.join(finetuned_bert2_dir,"eval_validation.pkl"), 'wb') as f:
        pickle.dump(results, f)
        
    #with open('saved_dictionary.pkl', 'rb') as f:
    #    loaded_dict = pickle.load(f)
    
    