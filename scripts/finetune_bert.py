from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TFAutoModelForTokenClassification, create_optimizer
from tensorflow.keras.callbacks import TensorBoard as TensorboardCallback
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd 
import ast, os


# set parameter: 
num_train_epochs = 20
train_batch_size = 128
num_warmup_steps = 0
eval_batch_size = 128
learning_rate = 2e-5
weight_decay_rate=0.01
output_dir = "/home/peetal/hulacon/street-element-extraction"
#tf.keras.mixed_precision.set_global_policy("mixed_float16")


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


if __name__ == "__main__": 
    
    # load dataset, 
    df_converters = {'tokens': ast.literal_eval, 'labels': ast.literal_eval}
    df = pd.read_csv("train_df_pretokenization.csv", converters=df_converters) 

    model_ckpt = "indobenchmark/indobert-base-p2" # specify model id 
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt) 

    # tokenization and align lables to sub-words
    df = df.rename(columns={"labels": "tags"})    
    ds = Dataset.from_pandas(df)
    ds_encoded = encode_dataset(ds)

    # split into training and validating 
    test_size=.15
    processed_dataset = ds_encoded.shuffle(seed=42).train_test_split(test_size=test_size)

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors="tf", padding = 'longest')

    # create tf datasets as model inputs 
    tf_train_dataset = processed_dataset['train'].to_tf_dataset(
        columns= ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        shuffle=False,
        batch_size=train_batch_size,
        collate_fn=data_collator
    )
    tf_val_dataset = processed_dataset['test'].to_tf_dataset(
        columns= ['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
        shuffle=False,
        batch_size=eval_batch_size,
        collate_fn=data_collator
    )

    # optimizer 
    num_train_steps = len(tf_train_dataset) * num_train_epochs
    #num_warmup_steps=int(0.1 * num_train_steps)
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=weight_decay_rate,
        num_warmup_steps=num_warmup_steps,
    )

    # config model 
    tag2index = {'B-POI':0, 'B-STR':1, 'E-POI':2, 'E-STR':3, 'I-POI':4, 'I-STR':5, 'S-POI':6, 'S-STR':7, 'O':8}
    index2tag = {y: x for x, y in tag2index.items()}
    model = TFAutoModelForTokenClassification.from_pretrained(
        model_ckpt,
        id2label=index2tag,
        label2id=tag2index
    )

    model.compile(optimizer=optimizer)

    # set call back
    callbacks=[]
    callbacks.append(TensorboardCallback(log_dir=os.path.join(output_dir,"logs")))
    callbacks.append(EarlyStopping(patience=2, restore_best_weights=True))
    
    
    with tf.device('/device:GPU:0'):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        model.fit(tf_train_dataset, validation_data=tf_val_dataset,callbacks=callbacks, epochs=num_train_epochs)

    model.save_pretrained("/home/peetal/hulacon/street-element-extraction/finetuned_bert2")

