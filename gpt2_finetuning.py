import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Load Q&A pairs from a file, tokenize them and prepare for training
def load_and_prepare_data(tokenizer, file_path='q&a_data.xlsx'):
    df = pd.read_excel(file_path)
    texts = df['Question'] + " " + df['Answer']
    inputs = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
    labels = inputs.input_ids.clone()
    # Shift the labels to the right
    labels[:, :-1] = labels[:, 1:].clone()
    labels[:, -1] = tokenizer.eos_token_id
    return Dataset.from_dict({'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'labels': labels})

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('woo_trained_model')
    model = GPT2LMHeadModel.from_pretrained('woo_trained_model')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_and_prepare_data(tokenizer)
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained('./woo_finetuned_model')
    tokenizer.save_pretrained('./woo_finetuned_model')

if __name__ == '__main__':
    main()










# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from datasets import Dataset
# import pandas as pd

# def load_qa_from_excel(excel_file_path):
#     # Load your dataset
#     df = pd.read_excel(excel_file_path)
#     questions_and_answers = []

#     for index, row in df.iterrows():
#         # Use the text directly without prompt engineering
#         questions_and_answers.append({'input_text': row['Question'] + " " + row['Answer']})

#     return questions_and_answers

# def main():
#     # Load tokenizer and model
#     tokenizer = GPT2Tokenizer.from_pretrained('woo_trained_model')
#     model = GPT2LMHeadModel.from_pretrained('woo_trained_model')  

#     # Ensure the pad_token is set in the tokenizer
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
#         model.resize_token_embeddings(len(tokenizer))

#     # Load Q&A pairs without prompt engineering
#     excel_file_path = 'q&apairs.xlsx'
#     questions_and_answers = load_qa_from_excel(excel_file_path)

#     # Prepare the dataset
#     inputs_and_labels = []
#     for qa in questions_and_answers:
#         # Tokenize the text
#         encoded_input = tokenizer(qa['input_text'], truncation=True, padding='max_length', max_length=512, return_tensors='pt')
#         # Create labels by shifting input IDs to the right
#         labels = torch.cat([encoded_input['input_ids'][:, 1:], torch.tensor([[tokenizer.eos_token_id]]).to(encoded_input['input_ids'].device)], dim=1)
#         inputs_and_labels.append({'input_ids': encoded_input['input_ids'].flatten(), 'attention_mask': encoded_input['attention_mask'].flatten(), 'labels': labels.flatten()})

#     # Convert to Dataset
#     dataset = Dataset.from_dict({k: [dic[k] for dic in inputs_and_labels] for k in inputs_and_labels[0]})
#     split_dataset = dataset.train_test_split(test_size=0.1)
#     train_dataset = split_dataset['train']
#     eval_dataset = split_dataset['test']
#     dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir='./results',
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=4,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir='./logs',
#         logging_steps=10,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False
#     )

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         tokenizer=tokenizer
#     )

#     # Train the model
#     trainer.train()

#     # Save the model
#     model.save_pretrained('woo_finetuned_model')
#     tokenizer.save_pretrained('woo_finetuned_model')

# if __name__ == '__main__':
#     main()
