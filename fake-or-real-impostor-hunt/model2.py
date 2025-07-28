from model import read_texts_from_dir
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

def main():
    # Load train and test data
    train_path = "./fake-or-real-impostor-hunt/dataset/train"
    df_train = read_texts_from_dir(train_path)
    test_path = "./fake-or-real-impostor-hunt/dataset/test"
    df_test = read_texts_from_dir(test_path)
    train_labels = pd.read_csv("./fake-or-real-impostor-hunt/dataset/train.csv", index_col="id")

    for i in range(df_train.shape[0]):
        if int(train_labels.loc[i, 'real_text_id']) == 2:
            # swap if real and fake are not in the place
            temp = df_train.iat[i, 0]
            df_train.iat[i, 0] = df_train.iat[i, 1]
            df_train.iat[i, 1] = temp

    df_train.columns = ['real', 'fake']
    df_train['label'] = 1  # real is label 1

    # mix real fake columns for train dataset
    real_part = pd.DataFrame({'value': df_train['real'], 'label': df_train['label']})
    fake_part = pd.DataFrame({'value': df_train['fake'], 'label': 0})
    mixed_df_train = pd.concat([real_part, fake_part], ignore_index=True)    

    # mix real fake columns for test dataset
    mixed_df_test = pd.concat([df_test['file_1'], df_test['file_2']], ignore_index=True)    
    mixed_df_test = pd.DataFrame(mixed_df_test, columns=['value'])

    # tokenized_test_1 = df_test['file_1'].map(lambda x: tokenizer(x, truncation=True, padding='max_length'))
    
    # # tokenized_test_2 = df_test['file_2'].map(lambda x: tokenizer_test(x, truncation=True, padding='max_length'))
    # tokenized_ids_1 = tokenized_test_1.map(lambda x: x['input_ids'])
    # # tokenized_ids_2 = tokenized_test_2.map(lambda x: x['input_ids'])

    # tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create huggingface dataset
    dataset_train = Dataset.from_pandas(mixed_df_train[['value', 'label']].rename(columns={'value': 'c'}))
    dataset_test = Dataset.from_pandas(mixed_df_test[['value']].rename(columns={'value': 'c'}))

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['c'], truncation=True, padding='max_length')

    tokenized_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_test = dataset_test.map(tokenize_function, batched=True)

    # Train/test split
    train_test_split = tokenized_train.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training
    training_args = TrainingArguments(output_dir="./fake-or-real-impostor-hunt/results", num_train_epochs=1)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    predict = trainer.predict(tokenized_test)
    
    logits = predict.predictions
    predicted_real_indices = []

    test_size = len(logits)//2

    for i in range(test_size):
        fake1, real1 = logits[i]
        fake2, real2 = logits[test_size + i]
        margin_1 = real1 - fake1
        margin_2 = real2 - fake2
        if margin_1 > margin_2:
            predicted_real_indices.append(1)
        else:
            predicted_real_indices.append(2)

    # To covert predicted_real_indices *list* to *dataframe*
    predicted_real_indices_df = pd.DataFrame({'real_text_id': predicted_real_indices})
    predicted_real_indices_df['id'] = predicted_real_indices_df.index
    predicted_real_indices_df = predicted_real_indices_df[['id', 'real_text_id']]
    
    # To write dataframe to csv file
    predicted_real_indices_df.to_csv('./fake-or-real-impostor-hunt/submission.csv', index=False)

if __name__ == "__main__":
    main()
