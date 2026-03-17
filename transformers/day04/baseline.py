from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

# --- Задача 1: Токенизация ---

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# Тест токенизации
sample = tokenize_texts(["Great movie!", "Terrible waste of time."])
print(f"Tokenized shape: {sample['input_ids'].shape}")
print()

# --- Задача 2: CLS-эмбеддинги ---

model = AutoModel.from_pretrained(model_name)
model.eval()


def get_cls_embeddings(texts, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        tokens = tokenize_texts(batch_texts)

        with torch.no_grad():
            outputs = model(**tokens)

        cls = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls.cpu().numpy())

    return np.vstack(all_embeddings)


# Тест эмбеддингов
emb = get_cls_embeddings(["Hello world", "Test sentence"])
print(f"Embeddings shape: {emb.shape}")
print()

# --- Задача 3: Logistic Regression на эмбеддингах ---

# Загружаем IMDB (берём подвыборку для скорости)
print("Loading IMDB dataset...")
dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(2000))

texts = dataset["text"]
labels = dataset["label"]  # 0 = negative, 1 = positive

print(f"Texts: {len(texts)}, Labels: {len(labels)}")
print(f"Class distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
print()

# Получаем эмбеддинги
print("Extracting embeddings (this may take a few minutes)...")
embeddings = get_cls_embeddings(texts, batch_size=16)
print(f"Embeddings shape: {embeddings.shape}")
print()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print()

# Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Результаты
report = classification_report(y_test, y_pred, target_names=["negative", "positive"])
f1 = f1_score(y_test, y_pred, average="macro")

print("Classification Report:")
print(report)
print(f"Macro F1: {f1:.4f}")

# Сохраняем результаты
with open("transformers/day04/baseline_results.txt", "w") as f:
    f.write("Baseline: DistilBERT embeddings + Logistic Regression\n")
    f.write(f"Dataset: IMDB (2000 samples)\n")
    f.write(f"Train: {len(X_train)}, Test: {len(X_test)}\n\n")
    f.write(report)
    f.write(f"\nMacro F1: {f1:.4f}\n")

print("\nResults saved to transformers/day04/baseline_results.txt")
