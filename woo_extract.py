import os
from tqdm import tqdm

def txt_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = r"C:\Users\Kshitijj\Downloads\wizardofoz"
output_file_train = "woo_output_train.txt"
output_file_val = "woo_output_val.txt"
vocab_file = "woo_vocab.txt"

files = txt_files_in_dir(folder_path)
total_files = len(files)

# Calculate the split indices
split_index = int(total_files * 0.9) # 90% for training
files_train = files[:split_index]
files_val = files[split_index:]

vocab = set()

with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_val, total=len(files_val)):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')
