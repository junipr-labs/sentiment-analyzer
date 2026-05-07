# Sentiment Analyzer

Batch text sentiment analysis with emotion detection, key phrase extraction, and language detection. Processes up to 100,000 texts using offline NLP — no external AI API needed, fast, deterministic, and private.

## Why Use This Actor?

| Feature | This Actor | Google NLP | AWS Comprehend | GPT-based actors |
|---------|-----------|------------|----------------|------------------|
| Price per 1K texts | $2.60 | $1.00-2.00 | $1.00 | $5.00-20.00 |
| External API needed | No | Yes | Yes | Yes |
| Deterministic results | Yes | Yes | Yes | No |
| Speed per text | < 5ms | ~100ms | ~100ms | ~1s |
| Emotion detection | Yes | Partial | No | Yes |
| Zero-config | Yes | Needs key | Needs key | Needs key |
| Data privacy | Full | External API | External API | External API |

At $2.60/1K texts, this actor is 3-13x cheaper than LLM-based alternatives while being 200x faster and fully deterministic. Your text data never leaves Apify infrastructure.

## How to Use

**Zero-config (just provide texts):**
```json
{
  "texts": [
    "This product is amazing! Best purchase ever.",
    "Terrible service, waited 2 hours.",
    "The package arrived on Tuesday."
  ]
}
```

**With ID and metadata:**
```json
{
  "texts": [
    { "id": "review-123", "text": "Love this product!", "metadata": { "source": "amazon" } }
  ]
}
```

**Review analysis pipeline (filter negative reviews):**
```json
{
  "texts": ["... your review texts ..."],
  "onlySentiment": "negative",
  "extractKeyPhrases": true
}
```

## Input Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `texts` | array | sample texts | Texts to analyze. Plain strings or `{id, text, metadata}` objects. Max 100K. |
| `sentimentModel` | string | `"combined"` | `afinn`, `vader`, or `combined` (most accurate) |
| `detectEmotions` | boolean | `true` | Detect joy, anger, sadness, fear, surprise, disgust |
| `extractKeyPhrases` | boolean | `true` | Extract key phrases contributing to sentiment |
| `detectLanguage` | boolean | `true` | Detect language of each text |
| `neutralThreshold` | number | `0.05` | Compound score range for neutral classification |
| `onlySentiment` | string | null | Filter: `positive`, `negative`, `neutral`, `mixed` |
| `includeWordScores` | boolean | `false` | Per-word sentiment scores |
| `minimumConfidence` | number | `0` | Minimum confidence threshold |

## Output Format

```json
{
  "id": "review-123",
  "text": "This product is amazing!",
  "textLength": 25,
  "language": { "detected": "en", "name": "English", "confidence": 0.9 },
  "sentiment": {
    "label": "positive",
    "compound": 0.8721,
    "positive": 0.8721,
    "negative": 0.0,
    "neutral": 0.1279,
    "confidence": 0.9721
  },
  "emotions": {
    "joy": 0.8, "anger": 0.0, "sadness": 0.0,
    "fear": 0.0, "surprise": 0.1, "disgust": 0.0,
    "dominant": "joy"
  },
  "keyPhrases": [
    { "phrase": "amazing", "sentiment": 1.0, "importance": 0.9 }
  ],
  "analyzedAt": "2026-03-11T12:00:00.000Z"
}
```

### Sentiment Score Interpretation

| Compound Score | Label | Description |
|----------------|-------|-------------|
| 0.05 to 1.0 | positive | Positive sentiment |
| -0.05 to 0.05 | neutral | No strong sentiment |
| -1.0 to -0.05 | negative | Negative sentiment |
| Both pos & neg > 0.3 | mixed | Contains both positive and negative |

## Tips and Advanced Usage

**Choosing the right model:**
- `combined` (default) — Best accuracy, uses AFINN + VADER ensemble
- `vader` — Better for social media with emojis, slang, caps, punctuation
- `afinn` — Fastest, purely lexicon-based

**Adjusting neutral threshold:**
- Default `0.05` works well for most use cases
- Increase to `0.15-0.20` for stricter positive/negative classification

**Handling sarcasm:** Lexicon-based analysis cannot reliably detect sarcasm ("Oh great, another delay"). For sarcasm, consider LLM-based tools.

**Multi-language:** Works best with English. 20+ languages supported via translated lexicons with lower accuracy.

## Pricing

**Pay-Per-Event:** $0.0026 per text analyzed ($2.60 per 1,000 texts)

Pricing includes all platform compute costs — no hidden fees.

| Scenario | Texts | Cost |
|----------|-------|------|
| Product reviews batch (500) | 500 | $1.30 |
| Twitter mentions (5,000) | 5,000 | $13.00 |
| Customer feedback (20,000) | 20,000 | $52.00 |
| Monthly monitoring (100,000) | 100,000 | $260.00 |

## Related Actors

- [Finance News Scraper](https://apify.com/junipr/finance-news-scraper) — Pair with sentiment for financial signal detection
- [Google News Scraper](https://apify.com/junipr/google-news-scraper) — News to analyze with sentiment

## FAQ

### How accurate is the sentiment analysis?
For standard review and social media text, lexicon-based approaches achieve 75-85% accuracy vs LLM methods (80-90%). For most business use cases, this difference is negligible at a 10x cost saving.

### Does it detect sarcasm?
No. Sarcasm detection requires contextual understanding that lexicon-based NLP cannot reliably provide. Document this limitation for your users.

### What languages are supported?
English is the primary language with highest accuracy. 20+ other languages are supported via translated lexicons with moderate accuracy.

### Can I analyze social media posts with emojis?
Yes — emoji sentiment mapping is built in. Common emojis are mapped to sentiment values.

### What's the difference between AFINN and VADER models?
AFINN: lexicon-based word scoring, fast and simple. VADER: rule-based with negation handling, emphasis (CAPS, !!!), and emoji support — better for social media.

### How does emotion detection work?
Emotion detection uses a word-emotion association lexicon. Each word is mapped to one of 6 emotions. The dominant emotion is the one most frequently detected in the text.
