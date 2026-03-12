from transformers import AutoTokenizer
import torch

# Загружаем токенизатор (мультиязычный — для русского и английского)
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Max length: {tokenizer.model_max_length}")
print()

# --- Токенизация одного текста ---
text = "This movie was absolutely amazing!"
tokens = tokenizer(text)
input_ids = tokens["input_ids"]

print(f"Текст: {text}")
print(f"Токены (IDs): {input_ids}")
print(f"Количество токенов: {len(input_ids)}")
print(f"Декодировано: {tokenizer.decode(input_ids)}")
print()


# --- Батч-токенизация ---
def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


texts = [
    "This movie was great!",
    "Terrible movie, waste of time.",
]

batch = tokenize_texts(texts)
print(f"Shape: {batch['input_ids'].shape}")
print(f"Attention mask:\n{batch['attention_mask']}")
print()

# --- Специальные токены ---
print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print()

single = tokenizer(text, return_tensors="pt")
print(f"Input IDs: {single['input_ids']}")
print(f"Decoded: {tokenizer.decode(single['input_ids'][0])}")
print()


# --- Детальный разбор токенизации ---
def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")


explain_tokenization("Transformers are amazing!", tokenizer)
