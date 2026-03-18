from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd
import torch

# --- Загрузка модели и данных ---

model_ft = AutoModelForSequenceClassification.from_pretrained(
    "transformers/day05/fine_tuned_model"
)
tokenizer = AutoTokenizer.from_pretrained("transformers/day05/fine_tuned_model")
model_ft.eval()

print("Loading IMDB dataset...")
dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(2000))

texts = dataset["text"]
labels = dataset["label"]

_, test_texts, _, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Задача 1: Получение предсказаний ---

predictions = []
probabilities = []

for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0][pred].item()

    predictions.append(pred)
    probabilities.append(conf)

# --- Задача 1: Анализ False Positive и False Negative ---

df_test = pd.DataFrame({
    "text": test_texts,
    "true_label": test_labels,
    "pred_label": predictions,
    "confidence": probabilities,
})

errors = df_test[df_test["true_label"] != df_test["pred_label"]]

fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]
fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

print(f"Всего тестовых примеров: {len(df_test)}")
print(f"Всего ошибок: {len(errors)} ({len(errors)/len(df_test)*100:.1f}%)")
print(f"False Positives: {len(fp)} (модель сказала positive, было negative)")
print(f"False Negatives: {len(fn)} (модель сказала negative, было positive)")

# --- Задача 2: Анализ паттернов ошибок ---

print("\n=== FALSE POSITIVES (модель сказала positive, было negative) ===")
for _, row in fp.head(5).iterrows():
    print(f"\n  Текст: {row['text'][:120]}...")
    print(f"  Уверенность: {row['confidence']:.1%}")

print("\n=== FALSE NEGATIVES (модель сказала negative, было positive) ===")
for _, row in fn.head(5).iterrows():
    print(f"\n  Текст: {row['text'][:120]}...")
    print(f"  Уверенность: {row['confidence']:.1%}")

# Анализ длины
df_test["text_length"] = df_test["text"].str.len()
errors["text_length"] = errors["text"].str.len()

avg_all = df_test["text_length"].mean()
avg_err = errors["text_length"].mean()
avg_correct = df_test[df_test["true_label"] == df_test["pred_label"]]["text_length"].mean()

print(f"\nСредняя длина всех текстов: {avg_all:.0f} символов")
print(f"Средняя длина ошибочных:    {avg_err:.0f} символов")
print(f"Средняя длина правильных:   {avg_correct:.0f} символов")

# Анализ уверенности
avg_conf_correct = df_test[df_test["true_label"] == df_test["pred_label"]]["confidence"].mean()
avg_conf_errors = errors["confidence"].mean()

print(f"\nСредняя уверенность (правильные): {avg_conf_correct:.1%}")
print(f"Средняя уверенность (ошибки):     {avg_conf_errors:.1%}")

# --- Задача 3: Сохранение анализа ---

with open("transformers/day07/error_analysis.txt", "w") as f:
    f.write("=== АНАЛИЗ ОШИБОК ===\n\n")
    f.write(f"Тестовых примеров: {len(df_test)}\n")
    f.write(f"Всего ошибок: {len(errors)} ({len(errors)/len(df_test)*100:.1f}%)\n")
    f.write(f"False Positives: {len(fp)}\n")
    f.write(f"False Negatives: {len(fn)}\n\n")

    f.write(f"Средняя длина всех текстов: {avg_all:.0f} символов\n")
    f.write(f"Средняя длина ошибочных:    {avg_err:.0f} символов\n")
    f.write(f"Средняя длина правильных:   {avg_correct:.0f} символов\n\n")

    f.write(f"Средняя уверенность (правильные): {avg_conf_correct:.1%}\n")
    f.write(f"Средняя уверенность (ошибки):     {avg_conf_errors:.1%}\n\n")

    f.write("=== ПРИМЕРЫ FALSE POSITIVES ===\n")
    for _, row in fp.head(5).iterrows():
        f.write(f"\nТекст: {row['text'][:200]}...\n")
        f.write(f"Уверенность: {row['confidence']:.1%}\n")

    f.write("\n\n=== ПРИМЕРЫ FALSE NEGATIVES ===\n")
    for _, row in fn.head(5).iterrows():
        f.write(f"\nТекст: {row['text'][:200]}...\n")
        f.write(f"Уверенность: {row['confidence']:.1%}\n")

    f.write("\n\n=== НАБЛЮДЕНИЯ ===\n")
    f.write("1. Ошибочные тексты в среднем длиннее — модель обрезает их до 64 токенов\n")
    f.write("   и теряет часть контекста (тональные слова могут быть в конце отзыва)\n")
    f.write("2. Модель менее уверена в ошибочных предсказаниях\n")
    f.write("3. Ирония и сарказм — частая причина ошибок (текст звучит positive,\n")
    f.write("   но смысл negative)\n")
    f.write("4. Смешанные отзывы ('фильм хороший, но концовка ужасная') сложны\n")
    f.write("   для бинарной классификации\n")

print("\nSaved: error_analysis.txt")
