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
    # df_train_real_fake = 

    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset('csv', data_files='your_data.csv')
    tokenized = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)

    # Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Training
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./results", evaluation_strategy="epoch"),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"]
    )
    trainer.train()

if __name__ == "__main__":
    main()
