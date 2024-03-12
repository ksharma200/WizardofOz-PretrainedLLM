import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Make sure to load both the model and tokenizer from the same directory
model_dir = './woo_trained_model'

# Load the trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

# Ensure the model is in evaluation mode
model.eval()

# Function to generate a response from a prompt
def generate_response(prompt_text):
    # Encode the prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    
    # Generate a sequence of tokens in response to the prompt
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=80,
        do_sample = True,
        temperature=0.01,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    
    # Decode the generated sequence to a string
    generated_sequence = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_sequence

# Interaction loop
while True:
    prompt = input("Prompt: ")
    if prompt.lower() == 'exit':
        break  # type 'exit' to end the loop
    completion = generate_response(prompt)
    print(f'Completion: {completion}')
