import re
import pandas as pd
from collections import Counter

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "]+",
    flags=re.UNICODE
)


total_mt_words, total_emojies, once_occured_count = 0, 0, 0
unique_words, unique_words_count, w_sorted, c_sorted = [], [], [], []

for chunk in pd.read_csv('./220642/dataset/qoura_questions.csv', chunksize=1):
    line = str(chunk.iloc[0]['question'])

    # mt words count
    words = line.strip().split()
    for w in words:
        if w not in unique_words:
            unique_words.append(w)
            unique_words_count.append(1)
        else:
            unique_words_count[unique_words.index(w)] += 1
            
        combined = list(zip(unique_words, unique_words_count))
        combined.sort()
        w_sorted, c_sorted = zip(*combined)

        wl = w.lower()
        if wl[0] == 'm' and wl[-1] == 't' and len(wl) > 4:
            total_mt_words += 1

    # emojies count
    found_emojis = emoji_pattern.findall(line)
    emoji_counts = Counter(found_emojis)
    total_emojies += sum(emoji_counts.values())

with open('./220642/output.txt', 'w') as f:
    f.write(str(total_mt_words) + "\n")
    f.write(str(total_emojies) + "\n")
    f.write(w_sorted[0] + ":" + str(c_sorted[0]) + " " 
            + w_sorted[1] + ":" + str(c_sorted[1]) + " " 
            + w_sorted[2] + ":" + str(c_sorted[2]) + " "
            + w_sorted[3] + ":" + str(c_sorted[3]) + " " 
            + w_sorted[4] + ":" + str(c_sorted[4]))
    f.write(str(sum(1 for c in unique_words_count if c==1)))


