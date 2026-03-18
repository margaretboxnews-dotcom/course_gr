# День 7 — Анализ ошибок и демо-приложение

## Зачем анализировать ошибки

Метрики (F1, accuracy) говорят "модель ошибается в 20% случаев", но не говорят
**почему**. Анализ ошибок помогает понять:

- Какие типы текстов модель не понимает
- Можно ли улучшить модель (и как)
- Стоит ли вообще использовать эту модель в продакшене

## False Positive vs False Negative

Для класса "positive":

- **False Positive (FP)** — модель сказала "positive", но на самом деле "negative".
  Модель "слишком оптимистична".

- **False Negative (FN)** — модель сказала "negative", но на самом деле "positive".
  Модель "слишком пессимистична".

Пример:
```
Текст: "I expected this movie to be great, but it was terrible."
Истинный класс: negative
Модель сказала: positive  ← False Positive

Почему ошибка? Начало текста positive ("expected to be great"),
и при обрезке до 64 токенов "terrible" могло не попасть.
```

## Типичные причины ошибок в нашем проекте

### 1. Обрезка текста (truncation)

Мы используем `max_length=64`, а IMDB-отзывы — длинные (200+ слов).
Модель видит только начало отзыва. Если тональное слово в конце — ошибка.

```
Полный отзыв: "The movie started slow, the plot seemed confusing,
              the acting was mediocre... but the ending was BRILLIANT!"
Модель видит: "The movie started slow, the plot seemed confusing..."
Предсказание: negative (а на самом деле positive)
```

### 2. Ирония и сарказм

```
"Oh sure, this is EXACTLY the kind of movie I wanted to waste 2 hours on."
Слова "exactly", "wanted" — positive по отдельности.
Но смысл — negative. Модель не улавливает сарказм.
```

### 3. Смешанные отзывы

```
"Great acting, beautiful cinematography, but the script was awful."
Тут и positive, и negative. Для бинарной классификации это сложно.
```

### 4. Малый объём обучающих данных

2000 примеров — мало для fine-tuning. С 25000 (полный IMDB) результаты
были бы значительно лучше.

## Уверенность модели (confidence)

```python
probs = torch.nn.functional.softmax(outputs.logits, dim=1)
```

softmax превращает "сырые" числа (logits) в вероятности:
```
logits = [-1.5, 2.3]  →  softmax  →  [0.02, 0.98]
                                       2% neg, 98% pos
```

Высокая уверенность (>90%) = модель "уверена" в ответе.
Низкая уверенность (50-60%) = модель "сомневается".

Ошибки чаще происходят при низкой уверенности — это можно использовать:
если модель не уверена, отправить текст на проверку человеку.

## Gradio — быстрое демо

Gradio создаёт веб-интерфейс из одной функции:

```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    ...

demo = gr.Interface(
    fn=predict_sentiment,      # функция предсказания
    inputs=gr.Textbox(...),    # поле ввода текста
    outputs=gr.Textbox(...),   # поле вывода результата
)
demo.launch()                  # запуск на http://127.0.0.1:7860
```

Gradio автоматически создаёт HTML-страницу с формой. Можно даже расшарить
ссылку (`demo.launch(share=True)`) — другие люди смогут протестировать модель.

## Итоги 7 дней

```
День 1: Текст → токены (числа)                    — токенизация
День 2: Токены → вектора (768D)                    — эмбеддинги
День 3: Визуализация attention                     — что видит модель
День 4: Вектора + LogReg = baseline (F1 ≈ 0.75)   — feature extraction
День 5: Fine-tuning всей модели (F1 ≈ 0.79)       — дообучение
День 6: Сравнение baseline vs fine-tuned           — метрики
День 7: Анализ ошибок + демо-приложение            — продакшен
```

Полный пайплайн NLP-проекта: от сырого текста до работающего приложения.
