import os
import random
import pandas as pd
import numpy as np
import time
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

# Function to read Q&A pairs from an Excel file
def load_qa_from_excel(excel_file_path):
    qa_pairs = []
    xls = pd.ExcelFile(excel_file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        for index, row in df.iterrows():
            question, answer = row['Question'], row['Answer']
            qa_pairs.append((question, answer))
    return qa_pairs

def main():
    # Load the pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Explicitly setting the pad token to eos_token (`<|endoftext|>`) for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Extract Q&A pairs
    excel_file_path = 'q&apairs.xlsx'
    questions_and_answers = load_qa_from_excel(excel_file_path)

    # Prepare the novel text and split it
    novel_filename = 'wizardofoz.txt'
    with open(novel_filename, 'r', encoding='utf-8') as file:
        novel_text = file.read()
        
    # Split the novel text into chunks and encode
    max_length = 1024 - tokenizer.num_special_tokens_to_add(pair=True)
    novel_chunks = [novel_text[i:i+max_length] for i in range(0, len(novel_text), max_length)]

    # Encode the Q&A pairs and novel text chunks
    combined_encodings = [tokenizer(text_pair[0], text_pair[1], truncation=True, max_length=max_length, padding="max_length")
                          for text_pair in questions_and_answers]
    combined_encodings += [tokenizer(chunk, truncation=True, max_length=max_length, padding="max_length") for chunk in novel_chunks]

    # Flatten the encodings
    input_ids = [encoding['input_ids'] for encoding in combined_encodings]
    attention_masks = [encoding['attention_mask'] for encoding in combined_encodings]

    # Split into training and validation datasets
    train_input_ids, val_input_ids, train_attention_masks, val_attention_masks = train_test_split(
        input_ids, attention_masks, test_size=0.15, random_state=42
    )

    # Convert the lists of lists into lists of tensors for compatibility with the Dataset object
    train_encodings = {'input_ids': train_input_ids, 'attention_mask': train_attention_masks}
    val_encodings = {'input_ids': val_input_ids, 'attention_mask': val_attention_masks}

    # Create Dataset objects for training and validation
    train_dataset = Dataset.from_dict(train_encodings)
    val_dataset = Dataset.from_dict(val_encodings)

    # Define the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    batch_sizes = [1]  # You may try larger if memory allows
    num_epochs = [3]  # Start with a single epoch for quick iteration
    learning_rates = [3e-4]  # Finer-grained learning rates
    num_workers = 4  # Number of data loader workers

    best_score = float('inf')
    best_params = {}

    # Loop over different parameters to find the best
    for batch_size in batch_sizes:
        for epoch in num_epochs:
            for lr in learning_rates:
                print(f"Training with batch size {batch_size}, {epoch} epoch(s), learning rate {lr}")

                # Define the training arguments with the current set of parameters
                training_args = TrainingArguments(
                    output_dir='./results',
                    overwrite_output_dir=True,
                    num_train_epochs=epoch,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    warmup_steps=500,
                    weight_decay=0.01,
                    save_steps=500,
                    learning_rate=lr,
                    evaluation_strategy='steps',
                    eval_steps=500,
                    logging_steps=10,
                    dataloader_num_workers=num_workers,
                    logging_dir='./logs',  # Directory for storing logs
                    load_best_model_at_end=True,  # Load the best model at the end of training
                    metric_for_best_model='loss',  # Use loss to determine the best model
                )

                # Initialize the Trainer with current parameters
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=lambda eval_pred: {"perplexity": np.exp(eval_pred[0])}  # Compute perplexity
                )

                # Train the model with the current parameters
                trainer.train()
                trainer.save_model()  # Save the model after training

                # Get evaluation metrics
                metrics = trainer.evaluate()
                perplexity = metrics['eval_perplexity']
                print(f"Perplexity: {perplexity}")

                # Choose best parameters based on perplexity
                if np.any(perplexity < best_score):
                    best_score = perplexity
                    best_params = {
                        'batch_size': batch_size,
                        'epoch': epoch,
                        'learning_rate': lr,
                        'perplexity': perplexity
                    }

    # Print out the best parameters after the loop
    print(f"Best parameters: {best_params}")

    # After training, save the model and tokenizer
    model.save_pretrained('./woo_trained_model')
    tokenizer.save_pretrained('./woo_trained_model')

if __name__ == '__main__':
    main()
