import tiktoken

# Load the GPT-2 tokenizer once globally
tokenizer = tiktoken.get_encoding("gpt2")

def encode(text: str) -> list[int]:
    return tokenizer.encode(text)

def decode(tokens: list[int]) -> str:
    return tokenizer.decode(tokens)