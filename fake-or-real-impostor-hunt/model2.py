from model import read_texts_from_dir, baseline_chars_method, baseline_method_english_word
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

def main():
    # Use the above function to load both train and test data
    train_path="./fake-or-real-impostor-hunt/dataset/train"
    df_train=read_texts_from_dir(train_path)
    test_path="./fake-or-real-impostor-hunt/dataset/test"
    df_test=read_texts_from_dir(test_path)
    train_labels = pd.read_csv("./fake-or-real-impostor-hunt/dataset/train.csv", index_col="id")

    for i in range(df_train.shape[0]):
        if int(train_labels.loc[i, 'real_text_id']) == 2:
            # swap if reak and fake are not in the place
            temp = df_train.iat[i, 0]
            df_train.iat[i, 0] = df_train.iat[i, 1]
            df_train.iat[i, 1] = temp

    df_train.columns = ['real', 'fake']

    # Tokenize trian
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # dataset = load_dataset('csv', data_files='your_data.csv')
    # tokenized = df_train.map(lambda x: tokenizer(x['real'], truncation=True, padding='max_length'), batched=True)
    tokenized_train = df_train['real'].map(lambda x: tokenizer(x, truncation=True, padding='max_length'))
    tokenized_ids_train = tokenized_train.map(lambda x: x['input_ids'])

    # Tokenize test
    tokenized_test_1 = df_test['file_1'].map(lambda x: tokenizer(x, truncation=True, padding='max_length'))
    # tokenized_test_2 = df_test['file_2'].map(lambda x: tokenizer_test(x, truncation=True, padding='max_length'))
    tokenized_ids_1 = tokenized_test_1.map(lambda x: x['input_ids'])
    # tokenized_ids_2 = tokenized_test_2.map(lambda x: x['input_ids'])


    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./results", evaluation_strategy="epoch"),
        train_dataset=tokenized_ids_train[:80],
        eval_dataset=tokenized_ids_train[80:]
    )
    trainer.train()
    output = trainer.predict(tokenized_ids_1)
    pass

if __name__ == "__main__":
    main()
