import random
import re
import torch
import argparse
from transformers import GPT2Tokenizer 
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import os

random.seed(42)
torch.manual_seed(42)


def parse_wet_file(file_path):
    """
    Parses the Common Crawl WET file and extracts text that is in English
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    documents = re.split(r'WARC/1.0', content)
    texts = []
    
    for doc in documents:
        if "WARC-Identified-Content-Language: eng" in doc:
            match = re.search(r'Content-Length: \d+\n\n(.*)', doc, re.DOTALL)
            if match:
                text = match.group(1)
                texts.append(text)
    
    return texts


def sample_random_prompts(texts, num_samples=25000, prompt_length=10):
    """
    Sample random prompts
    """
    prompts = []
    
    for text in texts:
        tokens = text.split()
        
        if len(tokens) >= prompt_length:
            start_index = random.randint(0, len(tokens) - prompt_length)
            prompt = tokens[start_index:start_index + prompt_length]
            prompts.append(' '.join(prompt))
        
        if len(prompts) >= num_samples:
            break
    
    return prompts


def generate_samples(wet_file_path, batch_size=50, num_samples=25000, max_length=256):
    
    texts = parse_wet_file(wet_file_path)
    prompts = sample_random_prompts(texts, num_samples)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    
    folder_path = "../models/XL/seed42_B"
    for ckpt in os.listdir(folder_path):
        print(f'processing : {ckpt}')
        
        model = AutoModelForCausalLM.from_pretrained(f"../models/XL/seed42_B/{ckpt}", return_dict=True, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16)
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        output_file = f"{ckpt}.txt"

        with open(output_file, "w", encoding="utf-8") as f:
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Samples"):
                batch_prompts = prompts[i:i+batch_size]
            
                inputs = tokenizer(
                batch_prompts, 
                return_tensors='pt', 
                truncation=True, 
                max_length=10  #prompt length should be 10 tokens 
                ).to(device)
            
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_length=max_length,
                        do_sample=False  # do_sample = True and top_k = 40 for top-k sampling 
                    )
            
                generations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
                for idx, generation in enumerate(generations):
                    f.write(f"{i+idx+1}. {generation}\n\n")

                    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wet_file", type=str, required=True, help="path to the common crawl WET file")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for generation")
    parser.add_argument("--num_samples", type=int, default=25000, help="Number of prompts")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    generate_samples(
        wet_file_path=args.wet_file,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        max_length=args.max_length
    )

if __name__ == "__main__":
    main()
