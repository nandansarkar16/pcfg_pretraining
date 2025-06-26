import glob
import os
from dataclasses import dataclass, field
from functools import partial

from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule
import pickle


@dataclass
class MatrixTextFiles(DataModule):

    train_data_path: Path
    val_data_path: Path

    seed: int = 42
    num_workers: int = 4

    tokenizer: Tokenizer = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        self.out_path_train = self.train_data_path / "train"
        self.out_path_val = self.val_data_path / "val"

    def connect(self, tokenizer: Tokenizer, batch_size: int = 1, max_seq_length: int = -1) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from litdata import optimize

        train_files = sorted(glob.glob(str(self.train_data_path / "*.pkl")))
        assert len(train_files) > 0, f"No .pkl files found in train data {train_files}"
        val_files = sorted(glob.glob(str(self.val_data_path / "*.txt")))
        assert len(val_files) > 0, f"No .txt files found in validation data {val_files}"

        num_workers = os.cpu_count() - 1
        use_workers = min(num_workers, len(train_files))
        if not Path(self.out_path_train).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize_and_store, tokenizer=self.tokenizer, max_seq_length=self.max_seq_length),
                inputs=train_files,
                output_dir=str(self.out_path_train),
                num_workers=use_workers,
                chunk_bytes="50MB",
            )
        
        use_workers = min(num_workers, len(val_files))
        if not Path(self.out_path_val).is_dir():
            validate_tokenizer(self.tokenizer)
            optimize(
                fn=partial(tokenize, tokenizer=self.tokenizer),
                inputs=val_files,
                output_dir=str(self.out_path_val),
                num_workers=use_workers,
                chunk_bytes="50MB",
            )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset

        train_dataset = StreamingDataset(
            input_dir=str(self.out_path_train),
            shuffle=True, #changed
            drop_last=True,
        )

        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, drop_last=True
        )
        return train_dataloader
    
    def val_dataloader(self) -> DataLoader:
        # Same as text files since val should just have tokens, and it is the wikipedia data
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=str(self.out_path_val),
            item_loader=TokensLoader(block_size=self.max_seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader

# default __get_item__ function from the dataset is what you yield here
def tokenize_and_store(filename: str, tokenizer: Tokenizer, max_seq_length: int):

    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    def add_matrices(matrix_1, matrix_2):
        rows_1, cols_1 = matrix_1.shape
        rows_2, cols_2 = matrix_2.shape

        new_matrix = torch.zeros(rows_1 + rows_2, cols_1 + cols_2, device=matrix_1.device)
        new_matrix[:rows_1, :cols_1] = matrix_1
        new_matrix[rows_1:, cols_1:] = matrix_2.to(matrix_1.device)

        return new_matrix
    
    data['distance_matrices'] = []
    data['tokens'] = []
    data['token_lengths'] = []
    i = 0
    # Tokenize text and create distance matrices for each sequence
    while i < len(data['words']):
        text = data['words'][i].strip() # must manually add newline token at the end since split() removes it
        tree = data['predicted_trees'][i]
        matrix = create_distance_matrix(text.split(), tree, tokenizer) # i add new line matrix row and column in create_distance_matrix
        T = 1 # Temperature parameter for the softmax
        matrix = matrix / T
        supervised_attention_matrix = create_supervised_attention_matrix(matrix)
        tokens = tokenizer.encode(text, bos=False, eos=False)
        new_line_token = torch.tensor([198]) # New line token
        tokens = torch.cat([tokens, new_line_token]) # Add newline token at the end
        data['tokens'].append(tokens)
        data['token_lengths'].append(len(tokens))
        data['distance_matrices'].append(supervised_attention_matrix)
        # delete below later 
        if i % 1000 == 0:
            print(f"print {i}", flush=True)  

        i += 1


    # Create samples
    samples = []
    i = 0
    while i < len(data['words']):
        sequence_tokens = data['tokens'][i]
        sequence_matrix = data['distance_matrices'][i]
        total_tokens_in_sequence = data['token_lengths'][i]

        # if the sequence is too long, split it into multiple sequences
        if total_tokens_in_sequence > max_seq_length:
            sequence_tokens = sequence_tokens.clone()[:max_seq_length]
            sequence_matrix = sequence_matrix.clone()[:max_seq_length, :max_seq_length]
            total_tokens_in_sequence = max_seq_length
            data['tokens'][i] = data['tokens'][i][max_seq_length:].clone()
            data['distance_matrices'][i] = data['distance_matrices'][i][max_seq_length:, max_seq_length:].clone()
            data['token_lengths'][i] = len(data['tokens'][i])
            i -= 1


        while total_tokens_in_sequence < max_seq_length:
            i += 1
            if i >= len(data['words']):
                break 

            next_tokens = data['tokens'][i]
            next_tokens_length = data['token_lengths'][i]
            next_matrix = data['distance_matrices'][i]

            if total_tokens_in_sequence + next_tokens_length > max_seq_length:
                # add part of it
                tokens_to_add = max_seq_length - total_tokens_in_sequence
                new_next_tokens = next_tokens.clone()[:tokens_to_add]
                new_next_matrix = next_matrix.clone()[:tokens_to_add, :tokens_to_add]
                data['tokens'][i] = next_tokens[tokens_to_add:] 
                data['distance_matrices'][i] = next_matrix[tokens_to_add:, tokens_to_add:] 
                data['token_lengths'][i] = len(data['tokens'][i])
                next_tokens = new_next_tokens
                next_matrix = new_next_matrix
                i -= 1

            sequence_tokens = torch.cat([sequence_tokens, next_tokens])
            sequence_matrix = add_matrices(sequence_matrix, next_matrix)
            total_tokens_in_sequence += next_tokens_length
        # Create the sample and store it
        # discard sequences too short (i.e. last sequence)
        if sequence_tokens.size()[0] == max_seq_length:
            # only take the first 1024 tokens since same dimension as attention matrix, since input ids are 1024 and targets are from 1 to 1025
            sequence_matrix = sequence_matrix[:max_seq_length - 1, :max_seq_length - 1]
            samples.append({
                    'tokens': sequence_tokens,  
                    'matrix': sequence_matrix
            })
            
        
        if i % 1000 == 0:
            print(f"sample {i}", flush=True)  

        i += 1

    for sample in samples:
        yield sample

# Tokenize the validation text in the same way as the training text
def tokenize(filename: str, tokenizer: Tokenizer):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    text = text.strip()
    yield tokenizer.encode(text, bos=True, eos=False)

    
def validate_tokenizer(tokenizer: Tokenizer) -> None:
    if tokenizer is None:
        raise ValueError(
            "Tokenizer is None. If you are using this data module via `litgpt pretrain`, "
            "please provide a valid `--tokenizer_dir` path."
        )

def lca(path1, path2):
    common_length = min(len(path1), len(path2))
    for i in range(common_length):
        if path1[i] != path2[i]:
            return i - 1  # LCA is the last common ancestor index
    return common_length - 1

def create_distance_matrix(words, tree_string, tokenizer):
    tree_tokens = tree_string.split()
    words_with_paths = []
    current_path = []
    token_index = 0

    for token in tree_tokens:
        if token == 'ı':
            current_path.append(token_index)
        elif token == '§':
            current_path.pop()
        else: # It's a word
            words_with_paths.append((token, current_path.copy()))
        token_index += 1


    # Make sure words have a space in front of them for correct tokenization, except for the first word
    # words = [word if i == 0 else " " + word for i, word in enumerate(words)] 
    # can comment out above line ^ since custom tokenizer has add_space enabled
    tokenized_words = [tokenizer.encode(word, bos=False, eos=False) for word in words]
    flattened_tokens = [word for sublist in tokenized_words for word in sublist]

    # Creating a map for each token to its word
    # keys are indicies in flattened_tokens, and values are the word that the token belongs to
    token_to_word = {}
    token_index = 0
    word_index = 0
    for word, tokens in zip(words, tokenized_words):
        for token in tokens:
            token_to_word[token_index] = (word, word_index)
            token_index += 1
        word_index += 1
    
    # Creating a distance matrix
    n = len(flattened_tokens)
    dist_matrix = -torch.ones((n + 1, n + 1), dtype=torch.int32) # n + 1 because we need to add the \n token (keeping it as -1)

    # Populate distance matrix
    for i in range(n):
        word_i, word_i_ind = token_to_word[i]
        for j in range(n):
            word_j, word_j_ind = token_to_word[j]
            if dist_matrix[i, j] == -1:
                if word_i == word_j and word_i_ind == word_j_ind:
                    # Tokens within the same word: distance is 1
                    dist = 1
                else:
                    path1 = words_with_paths[word_i_ind][1]
                    path2 = words_with_paths[word_j_ind][1]
                    # Find the LCA depth
                    lca_depth = lca(path1, path2)
                    # Distance is the sum of steps up from both words to their LCA, then down
                    dist = (len(path1) - lca_depth) + (len(path2) - lca_depth)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
    
    return dist_matrix

def create_supervised_attention_matrix(dist_matrix):
    # Compute the supervised attention matrix
    supervised_attention_matrix = torch.zeros_like(dist_matrix, dtype=torch.float32)
    n = dist_matrix.shape[0]
    for i in range(n):
        # For each row i, compute the attention weights for i >= j
        exp_vals = torch.exp(-dist_matrix[i, :i+1]) #* mask)  # Compute exp(-D_{i+1, j}) for j <= i
        sum_exp_vals = torch.sum(exp_vals)  # Sum of the exponential values for normalization
        supervised_attention_matrix[i, :i+1] = exp_vals / sum_exp_vals  # Normalize by the sum
    
    return supervised_attention_matrix