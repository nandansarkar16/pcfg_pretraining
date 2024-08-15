from nltk import Tree
import argparse
import pickle
from transformers import GPT2Tokenizer


def factorize(tree):
    def track(tree, i):
        label = tree.label()
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i+1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [[i, j, label]] + spans
        elif j > i:
            spans = [[i, j, 'NULL']] + spans
        return j, spans
    return track(tree, 0)[1]


# def create_dataset(file_name):
#     word_array = []
#     pos_array = []
#     gold_trees = []
#     with open(file_name, 'r') as f:
#         for line in f:
#             tree = Tree.fromstring(line)
#             token = tree.pos()
#             word, pos = zip(*token)
#             word_array.append(word)
#             pos_array.append(pos)
#             gold_trees.append(factorize(tree))
#     print("Dataset: ", word_array[0:3])
#     return {'word': word_array,
#             'pos': pos_array,
#             'gold_tree':gold_trees}


def create_dataset(file_name):
    max_len = 100
    token_count = 0
    line_count = 0
    word_array = []
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    with open(file_name, 'r') as f:
        for line in f:
            #tokens = tokenizer.tokenize(line)
            words = line.split()  
            if len(words) > max_len: 
                # num_breaks = len(tokens) // max_len  
                # for i in range(num_breaks):
                #     word_tuple = tuple(tokens[i * max_len:(i + 1) * max_len])
                #     word_array.append(word_tuple)
                # if len(tokens) % max_len != 0:  # Check if there are remaining tokens
                #     word_tuple = tuple(tokens[num_breaks * max_len:])
                #     word_array.append(word_tuple)
                continue
            else:
                word_tuple = tuple(words)
                word_array.append(word_tuple)
            token_count += len(words)
            line_count += 1
    print(f"Word count for {file_name}: {token_count}")
    print(f"Average word count per line for {file_name}: {token_count / line_count}")
    return {'word': word_array}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess ptb file.'
    )
    parser.add_argument('--train_file', default='data/ptb-train.txt')
    parser.add_argument('--val_file', default='data/ptb-valid.txt')
    parser.add_argument('--test_file', default='data/ptb-test.txt')
    parser.add_argument('--cache_path', default='data/wiki')

    args = parser.parse_args()

    result = create_dataset(args.train_file)
    with open(args.cache_path+"train.pickle", "wb") as f:
        pickle.dump(result, f)

    result = create_dataset(args.val_file)
    with open(args.cache_path+"val.pickle", "wb") as f:
        pickle.dump(result, f)

    result = create_dataset(args.test_file)
    with open(args.cache_path+"test.pickle", "wb") as f:
        pickle.dump(result, f)











