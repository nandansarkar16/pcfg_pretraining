import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import os
import json

class BLiMPDataset(Dataset):
    def __init__(self, blimp_data, max_length):
        self.blimp_data = blimp_data
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.blimp_data)
    
    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __getitem__(self, idx):
        sample = self.blimp_data[idx]
        sentence1 = sample['sentence_good']
        sentence2 = sample['sentence_bad']

        inputs1 = self.tokenizer(sentence1, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        inputs2 = self.tokenizer(sentence2, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)

        input_ids1 = inputs1['input_ids'].squeeze(0)
        input_ids2 = inputs2['input_ids'].squeeze(0)
        targets1 = input_ids1.clone()
        targets2 = input_ids2.clone()

        input_ids1 = input_ids1[:-1]
        input_ids2 = input_ids2[:-1]
        targets1 = targets1[1:]
        targets2 = targets2[1:]

        return input_ids1, targets1, input_ids2, targets2

def load_blimp_data(blimp_path):
    blimp_data = {}
    for filename in os.listdir(blimp_path):
        if filename.endswith(".jsonl"):
            with open(os.path.join(blimp_path, filename), 'r') as f:
                blimp_data[filename] = [json.loads(line) for line in f if json.loads(line)['field'] in ('syntax', 'syntax-semantics')]
            if len(blimp_data[filename]) == 0:
                del blimp_data[filename]
    return blimp_data

def evaluate_blimp_model(blimp_loader, model, device, pad_token_id):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for input_ids1, targets1, input_ids2, targets2 in blimp_loader:
            input_ids1, targets1 = input_ids1.to(device), targets1.to(device)
            input_ids2, targets2 = input_ids2.to(device), targets2.to(device)

            logits1 = model(input_ids1)
            logits2 = model(input_ids2)

            # Reshaping logits to [batch_size*seq_len, vocab_size] and targets to [batch_size*seq_len]
            logits1 = logits1.view(-1, logits1.size(-1))
            logits2 = logits2.view(-1, logits2.size(-1))
            targets1 = targets1.view(-1)
            targets2 = targets2.view(-1)

            loss1 = torch.nn.functional.cross_entropy(logits1, targets1, ignore_index=pad_token_id, reduction='none')
            loss2 = torch.nn.functional.cross_entropy(logits2, targets2, ignore_index=pad_token_id, reduction='none')

            # Reshaping targets back to [batch_size, seq_len]
            targets1 = targets1.view(input_ids1.size(0), -1)
            targets2 = targets2.view(input_ids2.size(0), -1)

            # Calculate sequence lengths by counting non-padding tokens
            sequence_length1 = (targets1 != pad_token_id).sum(dim=1)
            sequence_length2 = (targets2 != pad_token_id).sum(dim=1)

            # Sum the per-token losses
            loss1 = loss1.view(input_ids1.size(0), -1).sum(dim=1)
            loss2 = loss2.view(input_ids2.size(0), -1).sum(dim=1)

            # Average the summed losses by sequence length
            loss1 = loss1 / sequence_length1
            loss2 = loss2 / sequence_length2

            correct += (loss1 < loss2).sum().item()
            total += input_ids1.size(0)

    accuracy = correct / total
    return accuracy


def eval_all_blimp(model, max_seq_length):
    model.eval()
    dataset_path = '/data/cl/u/nsarkar/litgpt/litgpt/blimp_data'
    blimp_data = load_blimp_data(dataset_path)
    max_length = max_seq_length
    batch_size = 32
    total_accuracy = 0
    accuracies = {}

    for filename, data in blimp_data.items():
        print(f"Evaluating {filename}...")
        blimp_dataset = BLiMPDataset(data, max_length)
        blimp_loader = DataLoader(blimp_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        accuracy = evaluate_blimp_model(blimp_loader, model, torch.device('cuda'), blimp_dataset.get_pad_token_id())
        test_name = "BLiMP: " + filename.split('.')[0]
        accuracies[test_name] = accuracy
        total_accuracy += accuracy
    
    average_accuracy = total_accuracy / len(blimp_data)
    accuracies['Average'] = average_accuracy
    return accuracies