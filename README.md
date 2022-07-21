## 1. Overview: The problem and solution

### The Problem

- For many e-commerce companies (e.g., Amazon), address related-information can be very informative and can be harnessed to build more accurate geocode. This can result faster and more efficient shipping systems. Here I work with the real world data provided by [Shopee](https://shopee.com/), the leading online shopping platform in southeast Asia. They are interested in POI (point of interest) and street name for each customer but address-related information they receive are usually unstructured, in free-text format. Here is an example (Note that we are working with Indonesian here)

<img src="Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/Screen_Shot_2022-07-19_at_7.47.53_PM.png" alt="drawing" width="600"/>

### The Solution

- To solve this problem, I fine-tuned Indonesian BERT on [huggingface transformer](https://huggingface.co/) to perform Name-Entity-Recognition (i.e., token classification). Here I summarized the major steps I took and each step is elaborated in following sections.
    - Used IOBES annotation scheme to label each word.
    - Tokenized text inputs and aligned labels using a [pre-trained BERT for Indonesian](https://huggingface.co/indobenchmark/indobert-base-p2).
    - Added a token classification head and fine-tune both body and head.
    - Make predictions on unlabeled unstructured address; and reconstruct words from tokens.

<img src="Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/Screen_Shot_2022-07-19_at_9.17.13_PM.png" alt="drawing" width="700"/>

![ezgif.com-gif-maker.gif](Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/ezgif.com-gif-maker.gif)

### Data Preprocessing and Model Inputs

- Given raw, unstructured address: jalan tipar cakung no 26 depan rusun albo garasi dumtruk
    - ***Tokens***: using pre-trained tokenizer, the raw address was separated into sub-word tokens. [CLS] and [SEP] are also automatically added to the start and end of the sequence.
    - ***Tokens_ID***: INTs that map each token to the vocabulary of the pre-trained model
    - ***Words IDs***: specify which word the token belongs to. For example, both the token `tip` and the token `##ar` have the ID being 1, suggesting that these two tokens are in fact from the same word, and the word is the second in the sequence.
    - ***Labels:*** specify the name entity of each token. For example, the token `jalan` has the label `B-STR`, suggesting that it is the beginning of a street entity; the token `##o` has the label `E-POI`, suggesting that it is the end of a POI entity.
    - ***Labels_id***: category coding for all labels.
- The model was training with mini-batch gradient descent. Each mini-batch was prefetched and padded to the longest sequence of the batch using data collator.
- The inputs of the model are `Tokens_ID` and `Labels_id`, along with the `attention mask` for each sequence.

<img src="Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/Screen_Shot_2022-07-20_at_11.02.48_AM.png" alt="drawing" width="700"/>

### Model Training and Evaluating

- The model was trained ADAM with additional learning rate decay.
- With more training epoch, the training loss keeps dropping but the validation loss ended up getting larger, indicating a trend of overfitting. Thus, the model was restored to the weights trained after the third epoch.

<img src="Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/Screen_Shot_2022-07-20_at_11.19.36_AM.png" alt="drawing" width="700"/>

- The final model was evaluated using the validation set and the f1 score was computed for each tag category (POI and STR) using seqeval. The results show that the model was able to predict token classification with really good performance.

<img src="Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/Screen_Shot_2022-07-20_at_2.00.09_PM.png" alt="drawing" width="700"/>

### Model For Prediction

- It is shown below that before fine-tuning the model, the model is making random predictions for the labels of the tokens, with pretty high loss. However, after tuning, the model could accurately predict each token’s label. Note that loss was computed based on each token’s logit. Thus although the predicted label is correct, loss may still not be zero.

<img src="Fine-Tune%20BERT%20for%20Address%20Element%20Extraction%203f9f41c9e81c44b7812e67b9379bc278/Screen_Shot_2022-07-20_at_3.01.31_PM.png" alt="drawing" width="700"/>