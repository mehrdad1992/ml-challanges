import pandas as pd

stop_words = []
with open('./220641/dataset/stopwords.txt', encoding='utf-8') as f:
    for line in f:
        stop_words += line.strip()

total_uniques = 0
total_digits_question = 0
for chunk in pd.read_csv('./220641/dataset/qoura_questions.csv', chunksize=1):
    line = str(chunk.iloc[0]['question'])
    line_digits = line.strip()

    words = line.strip().split()
    unique_words = set(words)
    total_uniques += len(unique_words)

    digit_count = sum(1 for char in line_digits if char.isdigit())
    total_digits_question += digit_count


total_digits_shereno = 0
total_stop_words = 0
for chunk in pd.read_csv('./220641/dataset/shereno.csv', chunksize=1):
    line = str(chunk.iloc[0]['Poem']).strip()
    digit_count = sum(1 for char in line if char.isdigit())
    total_digits_shereno += digit_count

    words = sum(1 for word in line if word in stop_words)
    total_stop_words += words

with open('./220641/output.txt', 'w') as f:
    f.write(str(total_uniques) + "\n")
    f.write(str(total_digits_question) + " " + str(total_digits_shereno) + "\n")
    f.write(str(total_stop_words) + "\n")

