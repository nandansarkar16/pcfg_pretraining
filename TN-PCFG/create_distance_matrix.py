import torch

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
    
    #log.info(f"Words with paths: {words_with_paths}")


    tokenized_words = [tokenizer.tokenize(word) for word in words]
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
    dist_matrix = -torch.ones((n, n), dtype=torch.int32)

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

    # print(f"Distance matrix (unweighted): {dist_matrix}")
    # CREATE SUPERVISED ATTENTION MATRIX
    # (probably will move this to litgpt code)
    supervised_attention_matrix = torch.zeros_like(dist_matrix, dtype=torch.float32)
    supervised_attention_matrix[0, 0] = 1.0  # First token attends to itself
    n = dist_matrix.shape[0]
    for i in range(n):
        # For each row i, compute the attention weights for i >= j
        if i > 0:
            exp_vals = torch.exp(-dist_matrix[i, :i+1])  # Compute exp(-D_{i+1, j}) for j <= i
            sum_exp_vals = torch.sum(exp_vals)  # Sum of the exponential values for normalization
            supervised_attention_matrix[i, :i+1] = exp_vals / sum_exp_vals  # Normalize by the sum
    
    return supervised_attention_matrix