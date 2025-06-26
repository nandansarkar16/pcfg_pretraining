from transformers import GPT2TokenizerFast

# Load your custom tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("./saved_tokenizer")

special_tokens_dict = {
    "additional_special_tokens": ["__TEXT__", "__TREE__"]
}

num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)


tokenizer.save_pretrained("./saved_tokenizer_special_tokens", legacy_format=False)

print(tokenizer.convert_tokens_to_ids(["__TEXT__", "__TREE__"]))
