import os
import pickle
import tiktoken
import numpy as np
import random

# Ref: https://juejin.cn/post/7213945427853672507
# cat *.txt > input.txt
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

entries = []
with open(input_file_path, "r") as f:
    for line in f:
        if line.strip() and len(line) > 2:
            entries.append(line)

print(f"len of lines: {len(entries)}")
# Shuffle entries
random.shuffle(entries)

n = len(entries)
train_entries = entries[: int(n * 0.9)]
val_entries = entries[int(n * 0.9):]

# Turn those into strings
train_data = " ".join("{}".format(entry) for entry in train_entries)
val_data = " ".join("{}".format(entry) for entry in val_entries)

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))
