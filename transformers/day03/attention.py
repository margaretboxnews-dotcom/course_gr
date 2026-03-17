import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

# --- Загрузка токенизатора и модели с attention ---
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()

# --- Токенизация ---
text = "The amazing movie won many awards"
tokens = tokenizer(text, return_tensors="pt")

# --- Получение attention весов ---
with torch.no_grad():
    outputs = model(**tokens)

print(f"Количество слоёв: {len(outputs.attentions)}")
print(f"Форма attention для слоя 0: {outputs.attentions[0].shape}")

attention = outputs.attentions[0]
attn_single = attention[0, 0]
print(f"Single head shape: {attn_single.shape}")
print()


# --- Визуализация attention ---
def visualize_attention(tokens, attentions, layer=0, head=0, save_dir="."):
    attn = attentions[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap="viridis",
        cbar=True,
    )
    plt.title(f"Attention - Layer {layer}, Head {head}")
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attention_layer{layer}_head{head}.png")
    plt.close()
    print(f"Saved: attention_layer{layer}_head{head}.png")


# --- Attention из разных слоёв ---
visualize_attention(tokens, outputs.attentions, layer=0, head=0)
visualize_attention(tokens, outputs.attentions, layer=3, head=0)
visualize_attention(tokens, outputs.attentions, layer=5, head=0)

# --- Разные головы одного слоя ---
for head in range(8):
    visualize_attention(tokens, outputs.attentions, layer=0, head=head)

# --- Анализ внимания к ключевым словам ---
text_negative = "This movie was absolutely terrible and I hated it"
tokens_neg = tokenizer(text_negative, return_tensors="pt")

with torch.no_grad():
    outputs_neg = model(**tokens_neg)

visualize_attention(tokens_neg, outputs_neg.attentions, layer=5, head=0)

# --- Среднее attention по всем головам ---
token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
avg_attention = outputs.attentions[5][0].mean(dim=0)

plt.figure(figsize=(10, 8))
sns.heatmap(
    avg_attention.cpu().numpy(),
    xticklabels=token_list,
    yticklabels=token_list,
    cmap="viridis",
    cbar=True,
)
plt.title("Average Attention - Layer 5 (all heads)")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.tight_layout()
plt.savefig("attention_layer5_avg.png")
plt.close()
print("Saved: attention_layer5_avg.png")
