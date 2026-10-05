import 'dotenv/config';

const runtimeProcess = globalThis as typeof globalThis & {
  process?: {
    env?: Record<string, string | undefined>;
  };
};

const OLLAMA_HOST = runtimeProcess.process?.env?.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL = runtimeProcess.process?.env?.OLLAMA_MODEL || 'llama3.2:latest';
const LLM_DISABLED = typeof runtimeProcess.process?.env?.DISABLE_LLM === 'string'
  ? runtimeProcess.process?.env.DISABLE_LLM.toLowerCase() === 'true'
  : false;
let ollamaAvailable = !LLM_DISABLED;

interface OllamaResponse {
  model: string;
  response: string;
  done: boolean;
}

interface SymptomExtractionResult {
  symptoms: string[];
  severity?: string;
  duration?: string;
}

interface PredictionResult {
  disease: string;
  confidence: number;
  allPredictions: Array<{ disease: string; confidence: number }>;
  explanation: string;
}

function extractSymptomsLocally(userInput: string): SymptomExtractionResult {
  const text = userInput.toLowerCase();

  const symptomMap: Array<{ keywords: string[]; symptom: string }> = [
    { keywords: ['fever', 'temperature', 'hot body'], symptom: 'fever' },
    { keywords: ['cough', 'coughing'], symptom: 'cough' },
    { keywords: ['headache', 'head hurts', 'migraine'], symptom: 'headache' },
    { keywords: ['sore throat', 'throat pain'], symptom: 'sore throat' },
    { keywords: ['runny nose', 'stuffy nose', 'blocked nose', 'nasal congestion'], symptom: 'nasal congestion' },
    { keywords: ['fatigue', 'tired', 'weakness', 'weak'], symptom: 'fatigue' },
    { keywords: ['shortness of breath', 'breathing trouble', 'difficulty breathing', 'breathless'], symptom: 'shortness of breath' },
    { keywords: ['chest pain', 'chest tightness'], symptom: 'chest pain' },
    { keywords: ['nausea', 'nauseous'], symptom: 'nausea' },
    { keywords: ['vomit', 'vomiting'], symptom: 'vomiting' },
    { keywords: ['diarrhea', 'loose stool'], symptom: 'diarrhea' },
    { keywords: ['stomach pain', 'abdominal pain', 'stomach ache', 'abdomen pain'], symptom: 'abdominal pain' },
    { keywords: ['rash', 'skin rash', 'itching'], symptom: 'rash' },
    { keywords: ['dizzy', 'dizziness'], symptom: 'dizziness' },
    { keywords: ['body pain', 'body ache', 'muscle pain', 'muscle ache'], symptom: 'body aches' },
  ];

  const symptoms = symptomMap
    .filter(entry => entry.keywords.some(keyword => text.includes(keyword)))
    .map(entry => entry.symptom);

  const severity = text.includes('severe') || text.includes('intense')
    ? 'severe'
    : text.includes('mild')
      ? 'mild'
      : undefined;

  const durationMatch = userInput.match(/(\d+\s*(day|days|week|weeks|month|months|hour|hours))/i);
  const duration = durationMatch?.[1];

  return {
    symptoms: Array.from(new Set(symptoms)),
    severity,
    duration,
  };
}

function generateFallbackFollowUpQuestions(symptoms: string[]): string[] {
  const normalized = symptoms.map(symptom => symptom.toLowerCase());

  const questions: string[] = [
    'How long have you been having these symptoms?',
    'Are your symptoms getting better, worse, or staying the same?',
  ];

  if (normalized.some(symptom => symptom.includes('fever') || symptom.includes('temperature'))) {
    questions.push('What is the highest temperature you have recorded?');
  }

  if (normalized.some(symptom => symptom.includes('cough') || symptom.includes('breathing') || symptom.includes('shortness of breath'))) {
    questions.push('Are you having any trouble breathing or chest tightness?');
  }

  if (normalized.some(symptom => symptom.includes('pain') || symptom.includes('headache') || symptom.includes('ache'))) {
    questions.push('On a scale of 1 to 10, how severe is the pain or discomfort?');
  }

  if (normalized.some(symptom => symptom.includes('vomit') || symptom.includes('nausea') || symptom.includes('diarrhea') || symptom.includes('stomach'))) {
    questions.push('Have you had vomiting, diarrhea, or trouble keeping fluids down?');
  }

  if (normalized.some(symptom => symptom.includes('rash') || symptom.includes('skin'))) {
    questions.push('When did the rash or skin change start, and is it spreading?');
  }

  questions.push('Do you have any other symptoms that started around the same time?');

  return Array.from(new Set(questions)).slice(0, 5);
}

async function callOllama(prompt: string, systemPrompt?: string, timeoutMs = 100000): Promise<string> {
  if (LLM_DISABLED || !ollamaAvailable) {
    throw new Error('LLM is disabled or unavailable');
  }

  try {
    const body: Record<string, string | boolean> = {
      model: OLLAMA_MODEL,
      prompt,
      stream: false,
    };
    if (systemPrompt) {
      body.system = systemPrompt;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const response = await fetch(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const responseText = await response.text().catch(() => '');
      const isMemoryError = responseText.includes('requires more system memory')
        || responseText.includes('unable to allocate')
        || responseText.includes('out of memory')
        || responseText.includes('process has terminated')
        || responseText.includes('panic:');
      if (isMemoryError) {
        ollamaAvailable = false;
        throw new Error('Ollama model unavailable due to system memory constraints');
      }
      throw new Error(`Ollama error: ${response.status} ${responseText}`);
    }

    const data: OllamaResponse = await response.json();
    return data.response;
  } catch (error: any) {
    if (error.name === 'AbortError') {
      console.error(`Ollama call timed out after ${timeoutMs}ms`);
      throw new Error(`Ollama request timed out after ${timeoutMs}ms`);
    }
    console.error('Ollama call failed:', error);
    const msg = typeof error.message === 'string' ? error.message : '';
    if (msg.includes('memory') || msg.includes('unable to allocate') || msg.includes('terminated') || msg.includes('panic')) {
      ollamaAvailable = false;
    }
    throw error;
  }
}

export async function extractSymptoms(userInput: string): Promise<SymptomExtractionResult> {
  if (LLM_DISABLED || !ollamaAvailable) {
    console.warn('LLM unavailable: using local symptom extraction fallback');
    return extractSymptomsLocally(userInput);
  }

  const systemPrompt = `You are a medical symptom extractor. Analyze the user's description and extract all symptoms mentioned. Return a JSON array of symptoms found. Be specific and use medical terms where appropriate.

Examples:
- "I have a fever of 38°C" -> ["fever", "high temperature"]
- "My head hurts and I feel tired" -> ["headache", "fatigue"]
- "Chest pain when breathing" -> ["chest pain", "difficulty breathing"]

Return ONLY a JSON array of symptom strings, no other text.`;

  try {
    const result = await callOllama(
      `Extract symptoms from this patient description: "${userInput}"`,
      systemPrompt
    );

    try {
      // Clean markdown code blocks and trim
      let cleaned = result.replace(/^```(?:json)?\s*/m, '').replace(/\s*```$/m, '').trim();
      const symptoms = JSON.parse(cleaned);
      if (Array.isArray(symptoms)) {
        return { symptoms };
      }
      return extractSymptomsLocally(userInput);
    } catch {
      const lines = result.split('\n').filter(l => l.trim() && !l.includes('```'));
      return lines.length > 0 ? { symptoms: lines } : extractSymptomsLocally(userInput);
    }
  } catch (error) {
    console.warn('Falling back to local symptom extraction:', error);
    if (typeof error === 'object' && error && 'message' in error && String(error.message).includes('LLM is disabled')) {
      ollamaAvailable = false;
    }
    return extractSymptomsLocally(userInput);
  }
}

export async function generateFollowUpQuestions(symptoms: string[], context?: string): Promise<string[]> {
  if (LLM_DISABLED || !ollamaAvailable) {
    console.warn('LLM unavailable: using local follow-up question fallback');
    return generateFallbackFollowUpQuestions(symptoms);
  }

  const symptomsList = symptoms.join(', ');
  
  const systemPrompt = `You are a doctor asking YES/NO follow-up questions to understand a patient's condition. Generate 3-5 TARGETED YES/NO questions ONLY.

Rules (MANDATORY):
- ALL questions MUST be answerable with "yes" or "no"
- Focus: duration (e.g. "Have symptoms lasted >3 days?"), severity ("Is pain severe?"), triggers ("Worse at night?"), associated ("Any chest pain?"), risk ("Recent travel?"), recent changes ("Recent medication?")
- Medical, specific to symptoms
- No open-ended like "how long", "what temperature"

Return ONLY valid JSON array ["question1", "question2"], NO OTHER TEXT. Examples:
["Have symptoms lasted more than 3 days?", "Is the pain severe (8+ /10)?", "Any shortness of breath?", "Recent fever >38C?", "Worse when lying down?"]`;

  try {
    const result = await callOllama(
      `Generate follow-up questions for these symptoms: ${symptomsList}`,
      systemPrompt
    );

    try {
      // Clean markdown code blocks and trim
      let cleaned = result.replace(/^```(?:json)?\s*/m, '').replace(/\s*```$/m, '').trim();
      const questions = JSON.parse(cleaned);
      if (Array.isArray(questions)) {
        return questions.slice(0, 5);
      }
      return generateFallbackFollowUpQuestions(symptoms);
    } catch {
      const lines = result.split('\n').filter(l => l.trim() && !l.includes('```')).slice(0, 5);
      return lines.length > 0 ? lines : generateFallbackFollowUpQuestions(symptoms);
    }
  } catch (error) {
    console.warn('Falling back to local follow-up questions:', error);
    if (typeof error === 'object' && error && 'message' in error && String(error.message).includes('LLM is disabled')) {
      ollamaAvailable = false;
    }
    return generateFallbackFollowUpQuestions(symptoms);
  }
}

export async function generatePrediction(
  symptoms: string[],
  followUpAnswers: Array<{ question: string; answer: string }>
): Promise<PredictionResult> {
  const symptomsList = symptoms.join(', ');
  const answersList = followUpAnswers.map(a => `${a.question}: ${a.answer}`).join('; ');
  
  const context = `Patient symptoms: ${symptomsList}. Additional answers: ${answersList}`;

  const systemPrompt = `You are a medical diagnosis assistant. Based on the patient's symptoms and answers, provide a diagnosis with confidence score.

Medical knowledge to consider:
- Common symptom patterns for diseases
- Severity indicators
- Risk factors

Output format (JSON):
{
  "disease": "primary diagnosis",
  "confidence": 0-100,
  "allPredictions": [{"disease": "...", "confidence": 0-100}],
  "explanation": "brief explanation of the diagnosis"
}

Return ONLY valid JSON, no other text.`;

  try {
    const result = await callOllama(
      `Analyze and provide diagnosis for: ${context}`,
      systemPrompt
    );

    try {
      // Clean markdown code blocks and trim
      let cleaned = result.replace(/^```(?:json)?\s*/m, '').replace(/\s*```$/m, '').trim();
      const parsed = JSON.parse(cleaned);
      return {
        disease: parsed.disease || 'Unknown',
        confidence: parsed.confidence || 50,
        allPredictions: parsed.allPredictions || [{ disease: parsed.disease || 'Unknown', confidence: parsed.confidence || 50 }],
        explanation: parsed.explanation || 'Based on the symptoms provided.',
      };
    } catch {
      return {
        disease: 'Unable to determine',
        confidence: 0,
        allPredictions: [],
        explanation: 'Unable to analyze symptoms. Please consult a healthcare provider.',
      };
    }
  } catch (error) {
    console.error('Prediction generation error:', error);
    if (typeof error === 'object' && error && 'message' in error && String(error.message).includes('LLM is disabled')) {
      ollamaAvailable = false;
    }
    return {
      disease: 'Unable to determine',
      confidence: 0,
      allPredictions: [],
      explanation: 'Unable to analyze symptoms. Please consult a healthcare provider.',
    };
  }
}

export function generateFallbackExplanation(disease: string, symptoms: string[]): string {
  return `Why this appeared:\nThe symptoms you reported (${symptoms.join(', ')}) overlap with a reference pattern for "${disease}". This does not confirm that you have this condition.\n\n` +
    `What the score means:\nIt is a software pattern-matching score, not a probability, diagnosis, or measure of how likely you have this condition.\n\n` +
    `What to discuss with a professional:\nShare when your symptoms began, how they changed, your medical history, allergies, and current medicines with a qualified healthcare professional. Ask a clinician or pharmacist before starting, stopping, or changing medicines.\n\n` +
    `This AI-assisted screening is informational only and does not diagnose conditions or recommend treatment.`;
}

interface ExplanationMethodContext {
  predictionSource?: string;
  ragScore?: number;
  adjusted_ml?: number;
  mlGroup?: string;
  mlAgreementBoost?: number;
  final_confidence?: number;
  confidence_label?: string;
}

export async function generateExplanation(
  disease: string,
  symptoms: string[],
  methodCtx?: ExplanationMethodContext
): Promise<string> {
  if (LLM_DISABLED || !ollamaAvailable) {
    console.warn('LLM unavailable: using local explanation fallback');
    return generateFallbackExplanation(disease, symptoms);
  }

  // Build methodology summary for the LLM to include in its explanation
  let methodSummary = '';
  if (methodCtx) {
    const source = methodCtx.predictionSource ?? 'rag_primary';
    const rag = methodCtx.ragScore != null ? `${methodCtx.ragScore.toFixed(0)}% pattern score` : 'N/A';
    const conf = methodCtx.final_confidence != null ? `${methodCtx.final_confidence.toFixed(0)}% system score` : 'N/A';
    if (source === 'rag_primary') {
      methodSummary = `Symptom pattern matching (RAG): ${rag}. This score is not a probability. `;
      if (methodCtx.mlGroup) {
        methodSummary += `A separate model selected the broad "${methodCtx.mlGroup}" category. `;
      } else if (methodCtx.adjusted_ml && methodCtx.mlAgreementBoost) {
        methodSummary += 'A separate model agreed with the broad pattern. ';
      }
      methodSummary += `Final system score: ${conf}.`;
    } else if (source === 'ml_primary') {
      methodSummary = `ML ensemble score: ${(methodCtx.adjusted_ml ?? 0).toFixed(0)}%. RAG pattern score: ${rag}. Final system score: ${conf}.`;
    }
  }

  const systemPrompt = `You are a careful health-information explainer. You are not a clinician. This application produces a symptom-pattern comparison, not a diagnosis or a calibrated probability.

Structure your response with these exact sections using markdown **bold** for section headings:

**Why this appeared**
In 1-2 short sentences, say that the listed symptoms overlap with a reference pattern associated with "${disease}". Do not say the patient has or likely has this condition.

**What this score means**
Explain that the software score reflects pattern matching only; it is not a probability, diagnosis, or measure of likelihood. Use the methodology summary.

**What to discuss with a professional**
Suggest sharing symptom onset, changes, medical history, allergies, and current medicines with a qualified healthcare professional.

Keep the response concise, plain-language, and based only on supplied information. Do not invent patient history, provide a differential diagnosis, give emergency triage, or recommend, prescribe, start, stop, or change any treatment or medication.`;

  const userPrompt = `Reported symptoms: ${symptoms.join(', ')}.
Software's closest symptom-pattern match: "${disease}".
Methodology summary: ${methodSummary || 'symptom pattern matching analysis.'}

Write a patient-friendly explanation following your instructions.`;

  try {
    const result = await callOllama(userPrompt, systemPrompt);
    if (result && result.trim().length > 0) {
      return result.trim();
    }
    console.warn('Ollama returned empty explanation, using fallback');
    return generateFallbackExplanation(disease, symptoms);
  } catch (error) {
    console.error('Ollama explanation failed, using fallback:', error);
    if (typeof error === 'object' && error && 'message' in error && String(error.message).includes('LLM is disabled')) {
      ollamaAvailable = false;
    }
    return generateFallbackExplanation(disease, symptoms);
  }
}

export async function healthCheck(): Promise<boolean> {
  if (LLM_DISABLED || !ollamaAvailable) {
    return false;
  }

  try {
    const response = await fetch(`${OLLAMA_HOST}/api/tags`);
    if (!response.ok) {
      ollamaAvailable = false;
    }
    return response.ok;
  } catch {
    ollamaAvailable = false;
    return false;
  }
}

export function isLLMEnabled(): boolean {
  return !LLM_DISABLED && ollamaAvailable;
}
