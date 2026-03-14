import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

# --- Загрузка токенизатора и модели ---
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

print(model)
print()

# --- Hidden states для одного текста ---
text = "This movie was absolutely amazing!"
tokens = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**tokens)

print(f"Output type: {type(outputs)}")
print(f"Hidden states shape: {outputs.last_hidden_state.shape}")

cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"CLS embedding shape: {cls_embedding.shape}")
print(f"CLS embedding: {cls_embedding[0][:5]}...")
print()


# --- Функция для получения эмбеддингов ---
def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**tokens)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# --- Тест на нескольких текстах ---
texts = [
    "This movie was absolutely amazing!",
    "Terrible movie, waste of time.",
    "Pretty good, I liked it.",
    "Boring and too long.",
]

embeddings = get_embeddings(texts, tokenizer, model)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Ожидается: (4, 768) для DistilBERT")
print()


# --- Косинусное сходство ---
def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    return cosine_similarity(emb[0:1], emb[1:2])[0][0]


sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

print(f"Сходство похожих: {sim1:.3f}")
print(f"Сходство разных: {sim2:.3f}")
