from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import gradio as gr

# Загрузка модели
model = AutoModelForSequenceClassification.from_pretrained(
    "transformers/day05/fine_tuned_model"
)
tokenizer = AutoTokenizer.from_pretrained("transformers/day05/fine_tuned_model")
model.eval()

label_map = {0: "Negative", 1: "Positive"}


def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text."

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

    result = f"Prediction: {label_map[pred]}\n\n"
    result += "Probabilities:\n"
    for i, prob in enumerate(probs[0]):
        result += f"  {label_map[i]}: {prob * 100:.2f}%\n"

    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text for sentiment analysis..."),
    outputs=gr.Textbox(label="Result"),
    title="Sentiment Analysis with DistilBERT",
    description="Enter a movie review and the model will predict its sentiment.",
    examples=[
        ["This movie was absolutely fantastic!"],
        ["Terrible, waste of my time."],
        ["It was okay, nothing special."],
    ],
)

if __name__ == "__main__":
    demo.launch()
