from transformers import GPT2TokenizerFast

# Paths to your vocabulary and merges files
vocab_file = "./output_from_training/bpe-bytelevel-vocab.json"
merges_file = "./output_from_training/bpe-bytelevel-merges.txt"

# Initialize the GPT2Tokenizer with your custom vocabulary and merges
tokenizer = GPT2TokenizerFast(vocab_file=vocab_file, merges_file=merges_file)

# You can now use this tokenizer like any GPT-2 tokenizer
encoded_input = tokenizer("Example sentence for tokenization.")
print(encoded_input)

tokenizer.save_pretrained("./saved_tokenizer", legacy_format=False)

bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id

# Print the token IDs to verify
print("BOS Token ID:", bos_token_id)
print("EOS Token ID:", eos_token_id)