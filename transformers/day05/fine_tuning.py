from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import torch

# --- Задача 1: Dataset класс ---


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# --- Задача 2: Загрузка и подготовка данных ---

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading IMDB dataset...")
dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(2000))

texts = dataset["text"]
labels = dataset["label"]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=64)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=64)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# --- Задача 3: Модель и DataLoaders ---

num_labels = len(set(labels))
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# --- Задача 4: Настройка обучения ---

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Device: {device}")


# --- Задача 5: Функция обучения ---


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# --- Задача 6: Функция оценки ---


def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")

    return accuracy, f1


# --- Задача 7: Обучение ---

num_epochs = 3

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_acc, val_f1 = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    print(f"  Val F1: {val_f1:.4f}")
    print("-" * 50)

# --- Задача 8: Сохранение ---

save_dir = "transformers/day05/fine_tuned_model"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"\nModel saved to {save_dir}")

with open("transformers/day05/fine_tuned_results.txt", "w") as f:
    f.write("Fine-tuned DistilBERT on IMDB (2000 samples)\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n\n")
    f.write(f"Final Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Final Validation F1: {val_f1:.4f}\n")

print(f"Final Val F1: {val_f1:.4f} (baseline was 0.8175)")
