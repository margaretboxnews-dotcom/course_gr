from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# --- Задача 1: Загрузка моделей ---

model_name = "distilbert-base-uncased"

# Fine-tuned модель (День 5)
model_ft = AutoModelForSequenceClassification.from_pretrained(
    "transformers/day05/fine_tuned_model"
)
tokenizer = AutoTokenizer.from_pretrained("transformers/day05/fine_tuned_model")
model_ft.eval()
print("Fine-tuned model loaded")

# Baseline: воссоздаём подход из Дня 4 (CLS-эмбеддинги + LogReg)
base_model = AutoModel.from_pretrained(model_name)
base_model.eval()
print("Baseline embedding model loaded")


# --- Задача 2: Функция предсказания для fine-tuned ---


def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]

    predictions = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        predictions.append({
            "text": text[:80],
            "prediction": pred,
            "probabilities": probs[0].cpu().numpy(),
        })

    return predictions


# --- Задача 3: Baseline — CLS-эмбеддинги + LogReg ---


def get_cls_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**tokens)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# --- Загрузка данных (тот же датасет, что в Днях 4-5) ---

print("\nLoading IMDB dataset...")
dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(2000))

texts = dataset["text"]
labels = dataset["label"]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")

# --- Baseline: обучаем LogReg на CLS-эмбеддингах ---

print("\nExtracting baseline embeddings...")
train_emb = get_cls_embeddings(train_texts, tokenizer, base_model)
test_emb = get_cls_embeddings(test_texts, tokenizer, base_model)

baseline_clf = LogisticRegression(max_iter=1000, n_jobs=-1)
baseline_clf.fit(train_emb, train_labels)
y_pred_base = baseline_clf.predict(test_emb)

print("Baseline trained")

# --- Fine-tuned: предсказания на тесте ---

print("Running fine-tuned predictions...")
preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer)
y_pred_ft = [p["prediction"] for p in preds_ft]

# --- Задача 4: Сравнение на примерах ---

example_texts = [
    "This movie was absolutely fantastic!",
    "Terrible, waste of my time.",
    "It was okay, nothing special.",
    "Best film I've seen this year!",
    "Boring and too long.",
]

label_names = {0: "negative", 1: "positive"}

print("\n=== Примеры предсказаний ===\n")
preds_ex_ft = predict_fine_tuned(example_texts, model_ft, tokenizer)

for p in preds_ex_ft:
    label = label_names[p["prediction"]]
    conf = max(p["probabilities"]) * 100
    print(f"  [{label:>8s} {conf:5.1f}%] {p['text']}")

# --- Задача 5: Confusion matrices ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, title in [
    (axes[0], y_pred_base, "Baseline (LogReg + CLS embeddings)"),
    (axes[1], y_pred_ft, "Fine-tuned DistilBERT"),
]:
    cm = confusion_matrix(test_labels, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("transformers/day06/confusion_matrices.png", dpi=150)
print("\nSaved: confusion_matrices.png")

# --- Задача 6: Сравнение метрик ---

f1_base = f1_score(test_labels, y_pred_base, average="macro")
acc_base = accuracy_score(test_labels, y_pred_base)
f1_ft = f1_score(test_labels, y_pred_ft, average="macro")
acc_ft = accuracy_score(test_labels, y_pred_ft)

print("\n=== Baseline Model ===")
print(classification_report(test_labels, y_pred_base, target_names=["negative", "positive"]))

print("=== Fine-tuned Model ===")
print(classification_report(test_labels, y_pred_ft, target_names=["negative", "positive"]))

print("=== Сравнение ===")
print(f"Baseline    — F1: {f1_base:.4f}, Accuracy: {acc_base:.4f}")
print(f"Fine-tuned  — F1: {f1_ft:.4f}, Accuracy: {acc_ft:.4f}")

diff = (f1_ft - f1_base) / f1_base * 100
print(f"Разница F1: {diff:+.2f}%")

# --- Задача 7: Сохранение результатов ---

with open("transformers/day06/comparison_results.txt", "w") as f:
    f.write("=== Сравнение моделей ===\n\n")
    f.write("Датасет: IMDB (2000 samples, 80/20 split)\n")
    f.write(f"max_length: 64 токенов\n\n")
    f.write("Baseline (LogReg + CLS embeddings):\n")
    f.write(f"  F1 (macro): {f1_base:.4f}\n")
    f.write(f"  Accuracy:   {acc_base:.4f}\n")
    f.write(f"\nFine-tuned DistilBERT (3 epochs):\n")
    f.write(f"  F1 (macro): {f1_ft:.4f}\n")
    f.write(f"  Accuracy:   {acc_ft:.4f}\n")
    f.write(f"\nРазница F1: {diff:+.2f}%\n")

print("\nSaved: comparison_results.txt")
