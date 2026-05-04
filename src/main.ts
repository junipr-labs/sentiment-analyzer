import { Actor } from 'apify';
import { franc } from 'franc';
import natural from 'natural';

const { SentimentAnalyzer, PorterStemmer, WordTokenizer, TfIdf } = natural;

interface TextInput {
  id?: string;
  text: string;
  metadata?: Record<string, unknown>;
}

interface Input {
  texts?: Array<string | TextInput | Record<string, unknown>>;
  detectEmotions?: boolean;
  extractKeyPhrases?: boolean;
  detectLanguage?: boolean;
  sentimentModel?: 'afinn' | 'vader' | 'combined';
  neutralThreshold?: number;
  includeWordScores?: boolean;
  onlySentiment?: 'positive' | 'negative' | 'neutral' | 'mixed' | null;
  minimumConfidence?: number;
}

// Emoji sentiment map
const EMOJI_SENTIMENT: Record<string, number> = {
  '😊': 2, '😃': 2, '😄': 2, '😁': 2, '😆': 2, '🎉': 3, '🎊': 2, '👍': 2, '❤️': 3,
  '😍': 3, '🥰': 3, '😘': 2, '🤩': 3, '😎': 2, '🙌': 2, '💯': 2, '✨': 1, '🌟': 2,
  '😢': -2, '😭': -3, '😠': -3, '😡': -3, '😤': -2, '👎': -2, '💔': -3, '😞': -2,
  '😟': -2, '😩': -2, '😫': -2, '🤬': -3, '😒': -1, '🙁': -1, '😔': -2,
  '😐': 0, '😑': 0, '🤔': 0, '😶': 0,
};

// Emotion lexicon (simplified)
const EMOTION_WORDS: Record<string, string> = {
  // joy
  happy: 'joy', happiness: 'joy', joy: 'joy', joyful: 'joy', wonderful: 'joy', fantastic: 'joy',
  amazing: 'joy', excellent: 'joy', love: 'joy', adore: 'joy', delighted: 'joy', excited: 'joy',
  great: 'joy', awesome: 'joy', brilliant: 'joy', superb: 'joy', perfect: 'joy', beautiful: 'joy',
  // anger
  angry: 'anger', anger: 'anger', furious: 'anger', rage: 'anger', hate: 'anger', outraged: 'anger',
  mad: 'anger', annoyed: 'anger', irritated: 'anger', infuriated: 'anger', hostile: 'anger',
  // sadness
  sad: 'sadness', sadness: 'sadness', sorrow: 'sadness', grief: 'sadness', depressed: 'sadness',
  unhappy: 'sadness', miserable: 'sadness', heartbroken: 'sadness', disappointed: 'sadness', lonely: 'sadness',
  // fear
  fear: 'fear', afraid: 'fear', scared: 'fear', terrified: 'fear', horrified: 'fear', anxious: 'fear',
  worried: 'fear', nervous: 'fear', panic: 'fear', frightened: 'fear',
  // surprise
  surprised: 'surprise', shocked: 'surprise', astonished: 'surprise', amazed: 'surprise',
  unexpected: 'surprise', unbelievable: 'surprise', incredible: 'surprise',
  // disgust
  disgusting: 'disgust', disgust: 'disgust', revolting: 'disgust', gross: 'disgust', nasty: 'disgust',
  awful: 'disgust', horrible: 'disgust', terrible: 'disgust', dreadful: 'disgust', appalling: 'disgust',
};

// VADER-like word lists
const VADER_POSITIVE = new Set([
  'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'superb', 'outstanding',
  'love', 'awesome', 'perfect', 'brilliant', 'exceptional', 'splendid', 'marvelous', 'happy',
  'delightful', 'enjoyable', 'pleasant', 'nice', 'fine', 'positive', 'best', 'beautiful',
  'incredible', 'impressive', 'remarkable', 'tremendous', 'magnificent', 'glorious', 'joy',
  'thrilled', 'pleased', 'satisfied', 'content', 'excited', 'glad', 'grateful', 'blessed',
  'recommend', 'helpful', 'useful', 'easy', 'simple', 'convenient', 'efficient', 'reliable',
]);

const VADER_NEGATIVE = new Set([
  'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'poor', 'worst', 'hate', 'dislike',
  'disappointing', 'disappointed', 'useless', 'worthless', 'waste', 'broken', 'failed',
  'disgusting', 'nasty', 'lousy', 'pathetic', 'ridiculous', 'absurd', 'horrible', 'ugly',
  'stupid', 'idiotic', 'incompetent', 'unreliable', 'dishonest', 'fraud', 'scam', 'fake',
  'slow', 'buggy', 'crash', 'error', 'wrong', 'incorrect', 'misleading', 'confusing',
]);

const NEGATORS = new Set(['not', 'no', 'never', 'neither', 'nor', 'nobody', 'nothing',
  'nowhere', 'neither', 'nor', "don't", "doesn't", "didn't", "won't", "wouldn't",
  "can't", "cannot", "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
  "hasn't", "haven't", "hadn't", 'without', 'lack', 'lacking']);

const INTENSIFIERS = new Set(['very', 'extremely', 'incredibly', 'absolutely', 'totally',
  'utterly', 'completely', 'really', 'so', 'super', 'quite', 'remarkably', 'exceptionally',
  'highly', 'deeply', 'terribly', 'awfully', 'dreadfully', 'horribly']);

interface SentimentScores {
  label: string;
  compound: number;
  positive: number;
  negative: number;
  neutral: number;
  confidence: number;
}

function preprocessText(text: string): string {
  // Strip HTML
  let cleaned = text.replace(/<[^>]+>/g, ' ');
  // Strip URLs
  cleaned = cleaned.replace(/https?:\/\/\S+/g, ' ');
  // Strip markdown code blocks
  cleaned = cleaned.replace(/```[\s\S]*?```/g, ' ');
  // Normalize whitespace
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  return cleaned;
}

function getEmojiScore(text: string): number {
  let score = 0;
  for (const [emoji, val] of Object.entries(EMOJI_SENTIMENT)) {
    if (text.includes(emoji)) score += val;
  }
  return score;
}

function analyzeAFINN(tokens: string[]): { score: number; wordScores: Array<{ word: string; score: number; contribution: string }> } {
  const analyzer = new SentimentAnalyzer('English', PorterStemmer, 'afinn');
  const wordScores: Array<{ word: string; score: number; contribution: string }> = [];

  // Analyze full token list
  const totalScore = analyzer.getSentiment(tokens) * tokens.length;

  // Per-word analysis
  for (const token of tokens) {
    const ws = analyzer.getSentiment([token]);
    if (ws !== 0) {
      wordScores.push({
        word: token,
        score: ws,
        contribution: ws > 0 ? 'positive' : ws < 0 ? 'negative' : 'neutral',
      });
    }
  }

  return { score: totalScore, wordScores };
}

function analyzeVADER(tokens: string[], originalText: string): number {
  let score = 0;
  const isAllCaps = originalText === originalText.toUpperCase() && originalText.length > 3;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i].toLowerCase();
    let wordScore = 0;

    if (VADER_POSITIVE.has(token)) wordScore = 2;
    else if (VADER_NEGATIVE.has(token)) wordScore = -2;

    if (wordScore !== 0) {
      // Check for negation in window (-3 words)
      let negated = false;
      for (let j = Math.max(0, i - 3); j < i; j++) {
        if (NEGATORS.has(tokens[j].toLowerCase())) {
          negated = true;
          break;
        }
      }
      if (negated) wordScore *= -0.74;

      // Check for intensifiers
      if (i > 0 && INTENSIFIERS.has(tokens[i - 1].toLowerCase())) {
        wordScore *= 1.3;
      }

      // CAPS boost
      if (isAllCaps || token === token.toUpperCase() && token.length > 1) {
        wordScore *= 1.2;
      }

      score += wordScore;
    }
  }

  // Add emoji contribution
  score += getEmojiScore(originalText) * 0.5;

  // Repeated punctuation
  const exclamationCount = (originalText.match(/!/g) || []).length;
  if (exclamationCount > 1) score += Math.min(exclamationCount * 0.3, 1.5);

  return score;
}

function normalizeToCombined(afinnScore: number, vaderScore: number, textLength: number): SentimentScores {
  // Normalize scores to -1..1 range
  const afinnNorm = textLength > 0 ? Math.max(-1, Math.min(1, afinnScore / Math.max(textLength * 0.5, 1))) : 0;
  const vaderNorm = Math.max(-1, Math.min(1, vaderScore / 10));

  // Ensemble
  const compound = (afinnNorm * 0.4 + vaderNorm * 0.6);
  const clampedCompound = Math.max(-1, Math.min(1, compound));

  const positiveScore = Math.max(0, clampedCompound);
  const negativeScore = Math.max(0, -clampedCompound);
  const neutralScore = 1 - Math.abs(clampedCompound);

  const confidence = Math.min(1, Math.abs(clampedCompound) + 0.1);

  return {
    compound: parseFloat(clampedCompound.toFixed(4)),
    positive: parseFloat(positiveScore.toFixed(4)),
    negative: parseFloat(negativeScore.toFixed(4)),
    neutral: parseFloat(neutralScore.toFixed(4)),
    confidence: parseFloat(confidence.toFixed(4)),
    label: '', // filled below
  };
}

function classifySentiment(scores: SentimentScores, neutralThreshold: number): string {
  const { compound, positive, negative } = scores;
  // Mixed: both positive and negative are strong
  if (positive > 0.3 && negative > 0.3) return 'mixed';
  if (compound >= neutralThreshold) return 'positive';
  if (compound <= -neutralThreshold) return 'negative';
  return 'neutral';
}

function detectEmotions(tokens: string[]): Record<string, unknown> {
  const emotionCounts: Record<string, number> = {
    joy: 0, anger: 0, sadness: 0, fear: 0, surprise: 0, disgust: 0,
  };

  for (const token of tokens) {
    const emotion = EMOTION_WORDS[token.toLowerCase()];
    if (emotion && emotion in emotionCounts) {
      emotionCounts[emotion]++;
    }
  }

  const total = Object.values(emotionCounts).reduce((a, b) => a + b, 0) || 1;
  const normalizedEmotions: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(emotionCounts)) {
    normalizedEmotions[k] = parseFloat((v / total).toFixed(4));
  }

  let dominant = 'neutral';
  let maxVal = 0;
  for (const [k, v] of Object.entries(emotionCounts)) {
    if (v > maxVal) { maxVal = v; dominant = k; }
  }

  return { ...normalizedEmotions, dominant };
}

function extractKeyPhrases(text: string, tokens: string[], sentimentScores: SentimentScores): Array<{ phrase: string; sentiment: number; importance: number }> {
  const tfidf = new TfIdf();
  tfidf.addDocument(text);

  const phrases: Array<{ phrase: string; sentiment: number; importance: number }> = [];
  const seen = new Set<string>();

  // Find significant bigrams and unigrams
  const analyzer = new SentimentAnalyzer('English', PorterStemmer, 'afinn');

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i].toLowerCase();
    if (token.length < 3) continue;

    // Check unigram
    const ws = analyzer.getSentiment([token]);
    if (Math.abs(ws) > 0 && !seen.has(token)) {
      seen.add(token);
      phrases.push({
        phrase: token,
        sentiment: parseFloat(ws.toFixed(4)),
        importance: parseFloat(Math.min(1, Math.abs(ws) / 3).toFixed(4)),
      });
    }

    // Check bigram
    if (i < tokens.length - 1) {
      const bigram = `${token} ${tokens[i + 1].toLowerCase()}`;
      if (!seen.has(bigram)) {
        const bigramScore = analyzer.getSentiment([token, tokens[i + 1]]);
        if (Math.abs(bigramScore) > 0) {
          seen.add(bigram);
          phrases.push({
            phrase: bigram,
            sentiment: parseFloat(bigramScore.toFixed(4)),
            importance: parseFloat(Math.min(1, Math.abs(bigramScore) / 3).toFixed(4)),
          });
        }
      }
    }
  }

  return phrases.sort((a, b) => Math.abs(b.sentiment) - Math.abs(a.sentiment)).slice(0, 10);
}

function detectLanguage(text: string): { detected: string | null; name: string | null; confidence: number } {
  if (!text || text.length < 10) {
    return { detected: null, name: null, confidence: 0 };
  }

  const langCode = franc(text, { minLength: 5 });

  if (langCode === 'und') {
    return { detected: null, name: null, confidence: 0.1 };
  }

  const LANG_NAMES: Record<string, string> = {
    eng: 'English', fra: 'French', deu: 'German', spa: 'Spanish', ita: 'Italian',
    por: 'Portuguese', rus: 'Russian', zho: 'Chinese', jpn: 'Japanese', kor: 'Korean',
    ara: 'Arabic', nld: 'Dutch', pol: 'Polish', swe: 'Swedish', nor: 'Norwegian',
    dan: 'Danish', fin: 'Finnish', tur: 'Turkish', vie: 'Vietnamese', hin: 'Hindi',
  };

  const ISO_MAP: Record<string, string> = {
    eng: 'en', fra: 'fr', deu: 'de', spa: 'es', ita: 'it', por: 'pt', rus: 'ru',
    zho: 'zh', jpn: 'ja', kor: 'ko', ara: 'ar', nld: 'nl', pol: 'pl', swe: 'sv',
    nor: 'no', dan: 'da', fin: 'fi', tur: 'tr', vie: 'vi', hin: 'hi',
  };

  return {
    detected: ISO_MAP[langCode] ?? langCode.substring(0, 2),
    name: LANG_NAMES[langCode] ?? langCode,
    confidence: 0.8,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function analyzeText(
  rawInput: string | Record<string, unknown>,
  input: Input,
  afinnAnalyzer: any,
  tokenizer: any,
): Record<string, unknown> | null {
  let textStr: string;
  let id: string | null = null;
  let metadata: Record<string, unknown> | null = null;

  if (typeof rawInput === 'string') {
    textStr = rawInput;
  } else if (rawInput && typeof rawInput === 'object') {
    const obj = rawInput as Record<string, unknown>;
    textStr = (obj.text as string) ?? '';
    id = (obj.id as string) ?? null;
    metadata = (obj.metadata as Record<string, unknown>) ?? null;
  } else {
    return null;
  }

  // Handle empty string
  if (textStr === '' || textStr == null) {
    return {
      id,
      text: '',
      textLength: 0,
      language: { detected: null, name: null, confidence: 0 },
      sentiment: { label: 'neutral', compound: 0, positive: 0, negative: 0, neutral: 1, confidence: 0 },
      emotions: input.detectEmotions ? { joy: 0, anger: 0, sadness: 0, fear: 0, surprise: 0, disgust: 0, dominant: 'neutral' } : undefined,
      keyPhrases: input.extractKeyPhrases ? [] : undefined,
      wordScores: input.includeWordScores ? [] : undefined,
      metadata,
      analyzedAt: new Date().toISOString(),
    };
  }

  // Truncate if too long
  let text = textStr;
  const truncated = text.length > 10000;
  if (truncated) {
    text = text.substring(0, 10000);
  }

  const cleaned = preprocessText(text);
  const tokens = tokenizer.tokenize(cleaned) ?? [];

  // Language detection
  const language = input.detectLanguage !== false ? detectLanguage(cleaned) : { detected: null, name: null, confidence: 0 };

  // Sentiment analysis
  const model = input.sentimentModel ?? 'combined';
  let sentimentScores: SentimentScores;
  let wordScores: Array<{ word: string; score: number; contribution: string }> = [];

  if (model === 'afinn') {
    const afinnResult = analyzeAFINN(tokens);
    wordScores = afinnResult.wordScores;
    const norm = normalizeToCombined(afinnResult.score, 0, tokens.length);
    sentimentScores = norm;
  } else if (model === 'vader') {
    const vaderScore = analyzeVADER(tokens, cleaned);
    sentimentScores = normalizeToCombined(0, vaderScore, tokens.length);
  } else {
    // combined
    const afinnResult = analyzeAFINN(tokens);
    wordScores = afinnResult.wordScores;
    const vaderScore = analyzeVADER(tokens, cleaned);
    sentimentScores = normalizeToCombined(afinnResult.score, vaderScore, tokens.length);
  }

  const neutralThreshold = input.neutralThreshold ?? 0.05;
  sentimentScores.label = classifySentiment(sentimentScores, neutralThreshold);

  // Filter by sentiment if requested
  if (input.onlySentiment && sentimentScores.label !== input.onlySentiment) {
    return null;
  }

  // Filter by minimum confidence
  if (input.minimumConfidence && sentimentScores.confidence < input.minimumConfidence) {
    return null;
  }

  // Emotions
  const emotions = input.detectEmotions !== false ? detectEmotions(tokens) : undefined;

  // Key phrases
  const keyPhrases = input.extractKeyPhrases !== false ? extractKeyPhrases(cleaned, tokens, sentimentScores) : undefined;

  const result: Record<string, unknown> = {
    id,
    text: textStr.substring(0, 500), // truncate for storage
    textLength: textStr.length,
    language,
    sentiment: sentimentScores,
    metadata,
    analyzedAt: new Date().toISOString(),
  };

  if (emotions !== undefined) result.emotions = emotions;
  if (keyPhrases !== undefined) result.keyPhrases = keyPhrases;
  if (input.includeWordScores) result.wordScores = wordScores;
  if (truncated) result.errors = ['TEXT_TRUNCATED'];

  return result;
}

async function main() {
  await Actor.init();

  const input = (await Actor.getInput<Input>()) ?? {};
  const {
    texts = [
      "This product is amazing! Best purchase I've ever made.",
      'Terrible service, waited 2 hours and nobody helped.',
      'The package arrived on Tuesday as expected.',
    ],
    sentimentModel = 'combined',
    detectEmotions: doDetectEmotions = true,
    extractKeyPhrases: doExtractKeyPhrases = true,
    detectLanguage: doDetectLanguage = true,
  } = input;

  if (texts.length > 100000) {
    await Actor.pushData({
      error: true,
      code: 'BATCH_TOO_LARGE',
      message: 'Input exceeds 100,000 texts limit.',
    });
    await Actor.exit('Batch too large');
    return;
  }

  const tokenizer = new WordTokenizer();
  const afinnAnalyzer = new SentimentAnalyzer('English', PorterStemmer, 'afinn');

  const stats = {
    totalAnalyzed: 0,
    totalErrors: 0,
    sentimentBreakdown: { positive: 0, negative: 0, neutral: 0, mixed: 0 },
    emotionBreakdown: { joy: 0, anger: 0, sadness: 0, fear: 0, surprise: 0, disgust: 0 },
    languageBreakdown: {} as Record<string, number>,
    compoundSum: 0,
    startTime: Date.now(),
  };

  // Process in chunks for large batches
  const CHUNK_SIZE = 5000;
  for (let i = 0; i < texts.length; i += CHUNK_SIZE) {
    const chunk = texts.slice(i, i + CHUNK_SIZE);
    const results: Record<string, unknown>[] = [];

    for (const textItem of chunk) {
      try {
        const result = analyzeText(textItem as string | Record<string, unknown>, input, afinnAnalyzer, tokenizer);
        if (!result) continue; // filtered out

        results.push(result);
        stats.totalAnalyzed++;

        const sentiment = result.sentiment as { label: string; compound: number };
        const label = sentiment.label as keyof typeof stats.sentimentBreakdown;
        if (label in stats.sentimentBreakdown) {
          stats.sentimentBreakdown[label]++;
        }
        stats.compoundSum += sentiment.compound;

        const lang = (result.language as { detected: string | null }).detected;
        if (lang) {
          stats.languageBreakdown[lang] = (stats.languageBreakdown[lang] ?? 0) + 1;
        }

        if (result.emotions) {
          const emotions = result.emotions as Record<string, number>;
          const dominant = (result.emotions as Record<string, unknown>).dominant as string;
          if (dominant && dominant in stats.emotionBreakdown) {
            stats.emotionBreakdown[dominant as keyof typeof stats.emotionBreakdown]++;
          }
        }
      } catch (e) {
        stats.totalErrors++;
        console.error('Text analysis error:', e);
      }
    }

    if (results.length > 0) {
      try {
        await Actor.charge({ eventName: 'text-analyzed', count: results.length });
      } catch { /* PPE not configured */ }
      await Actor.pushData(results);
    }
  }

  // Write run summary
  const durationMs = Date.now() - stats.startTime;
  const total = stats.totalAnalyzed;
  const summary = {
    totalAnalyzed: total,
    totalErrors: stats.totalErrors,
    sentimentBreakdown: stats.sentimentBreakdown,
    sentimentPercentages: {
      positive: total > 0 ? parseFloat((stats.sentimentBreakdown.positive / total * 100).toFixed(1)) : 0,
      negative: total > 0 ? parseFloat((stats.sentimentBreakdown.negative / total * 100).toFixed(1)) : 0,
      neutral: total > 0 ? parseFloat((stats.sentimentBreakdown.neutral / total * 100).toFixed(1)) : 0,
      mixed: total > 0 ? parseFloat((stats.sentimentBreakdown.mixed / total * 100).toFixed(1)) : 0,
    },
    averageCompound: total > 0 ? parseFloat((stats.compoundSum / total).toFixed(4)) : 0,
    emotionBreakdown: stats.emotionBreakdown,
    languageBreakdown: stats.languageBreakdown,
    durationMs,
    averageAnalysisMs: total > 0 ? parseFloat((durationMs / total).toFixed(2)) : 0,
  };

  const store = await Actor.openKeyValueStore();
  await store.setValue('OUTPUT', summary);

  console.log(`Sentiment analysis complete. Analyzed ${total} texts in ${durationMs}ms.`);
  await Actor.exit();
}

main().catch(async (err) => {
  console.error('Fatal error:', err);
  await Actor.exit('Fatal error: ' + err.message);
});
