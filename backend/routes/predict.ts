import { Router, Response } from 'express';
import { AuthRequest, authMiddleware } from '../middleware/auth.js';
import * as llmService from '../services/llm.service.js';
import * as ragService from '../services/rag.service.js';

const router = Router();
const USE_LLM = process.env.DISABLE_LLM?.toLowerCase() !== 'true';

// ML Service configuration
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// ── Stage 1: Symptom Preprocessing ────────────────────────────────────────

/**
 * Normalize symptom text: lowercase, strip punctuation/non-alphanumeric
 */
function normalizeSymptom(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim();
}

/**
 * Synonym expansion map – maps common user terms to standardized KB forms
 */
const SYMPTOM_SYNONYMS: Record<string, string[]> = {
  // ── General ──
  headache: ['cephalgia', 'head pain', 'head hurts', 'migraine', 'headaches'],
  fever: ['temperature', 'hot body', 'high temperature', 'feverish', 'high fever'],
  fatigue: ['tired', 'weakness', 'weak', 'extreme fatigue', 'tiredness', 'exhaustion'],
  cough: ['coughing', 'dry cough', 'wet cough', 'persistent cough'],
  nausea: ['nauseous', 'feeling sick', 'queasy'],
  vomiting: ['vomit', 'throwing up', 'puking'],
  diarrhea: ['loose stool', 'loose stools', 'watery stool', 'loose bowel'],
  'skin rash': ['rash', 'eruption', 'itchy rash', 'blisters', 'skin eruption'],
  // ── Respiratory ──
  'shortness of breath': ['breathing trouble', 'difficulty breathing', 'breathless', 'cant breathe', 'hard to breathe', 'laboured breathing'],
  'nasal congestion': ['stuffy nose', 'blocked nose', 'runny nose', 'runny nose'],
  'sore throat': ['throat pain', 'throat hurts', 'scratchy throat'],
  wheezing: ['wheeze', 'whistling breath'],
  // ── Chest / Cardiac ──
  'chest tightness': ['tight chest', 'chest pressure', 'chest heaviness'],
  'sharp chest pain': ['chest pain', 'chest hurts', 'pain in chest'],
  'palpitations': ['heart racing', 'racing heart', 'heart pounding', 'irregular heartbeat'],
  // ── GI ──
  'sharp abdominal pain': ['abdominal pain', 'stomach pain', 'stomach ache', 'abdomen pain', 'tummy pain', 'belly pain'],
  'lower abdominal pain': ['lower stomach pain', 'lower belly pain', 'pelvic discomfort'],
  'decreased appetite': ['loss of appetite', 'poor appetite', 'no appetite', 'not hungry', 'appetite loss', 'not eating'],
  'difficulty in swallowing': ['trouble swallowing', 'painful swallowing', 'hard to swallow', 'dysphagia'],
  constipation: ['constipated', 'cannot poop', 'no bowel movement'],
  'stomach bloating': ['bloating', 'bloated stomach', 'bloated belly', 'distended abdomen'],
  // ── Urinary ──
  'frequent urination': ['urinary frequency', 'frequent peeing', 'urge to urinate', 'urinary urgency', 'need to pee often', 'peeing a lot'],
  'painful urination': ['burning urination', 'burning sensation urinating', 'penile burning sensation', 'burning pee', 'stinging urination', 'hurts to urinate', 'dysuria', 'pain when urinating'],
  // ── Musculoskeletal ──
  'joint pain': ['joint ache', 'joint hurts', 'achy joints', 'painful joints'],
  'joint swelling': ['swollen joints', 'puffy joints', 'inflamed joints'],
  'back pain': ['back ache', 'back hurts', 'lower back pain', 'lumbar pain'],
  'muscle pain': ['muscle ache', 'muscle soreness', 'myalgia'],
  'ache all over': ['body aches', 'body pain', 'aching all over', 'generalized aching', 'body aches all over'],
  // ── Neurological ──
  dizziness: ['dizzy', 'lightheaded', 'vertigo', 'spinning sensation'],
  // ── Eyes ──
  'diminished vision': ['blurred vision', 'blurry vision', 'vision problems', 'blurry eyesight', 'poor vision', 'vision changes'],
  // ── Skin ──
  'itching of skin': ['itching', 'itchy skin', 'skin itching', 'pruritus', 'itchiness'],
  // ── Other ──
  'weight gain': ['gaining weight', 'putting on weight', 'increased weight'],
  'weight loss': ['losing weight', 'recent weight loss', 'unexplained weight loss'],
  thirst: ['excessive thirst', 'very thirsty', 'always thirsty', 'polydipsia'],
  insomnia: ['cant sleep', 'trouble sleeping', 'sleep problems', 'sleeplessness'],
  'loss of taste or smell': ['cant taste', 'no sense of smell', 'lost taste', 'anosmia', 'lost smell', 'no taste'],
  depression: ['depressed', 'feeling depressed', 'low mood', 'sad all the time'],
  'anxiety and nervousness': ['anxiety', 'anxious', 'nervous', 'feeling anxious', 'panic'],
  sweating: ['excessive sweating', 'night sweats', 'sweaty', 'profuse sweating'],
};

/**
 * Expand synonyms – maps any synonym to its canonical form
 */
function expandSynonyms(symptom: string): string {
  const normalized = normalizeSymptom(symptom);
  for (const [canonical, synonyms] of Object.entries(SYMPTOM_SYNONYMS)) {
    if (normalized === canonical || synonyms.some(s => normalizeSymptom(s) === normalized)) {
      return canonical;
    }
  }
  return normalized;
}

/**
 * Gate: minimum 3 symptoms required for reliable prediction
 */
function symptomGate(symptoms: string[]): { passed: boolean; error?: string } {
  if (symptoms.length < 3) {
    return {
      passed: false,
      error: 'MIN_SYMPTOMS',
    };
  }
  return { passed: true };
}

/**
 * Full preprocessing pipeline – returns normalized + expanded symptoms
 */
function preprocessSymptoms(rawSymptoms: string[]): { symptoms: string[]; gate: { passed: boolean; error?: string } } {
  const normalized = rawSymptoms.map(normalizeSymptom);
  const expanded = normalized.map(expandSynonyms);
  const unique = Array.from(new Set(expanded)).filter(s => s.length > 0);
  const gate = symptomGate(unique);
  return { symptoms: unique, gate };
}

// ── Stage 4: Confidence Label Mapping ─────────────────────────────────────

function getConfidenceLabel(score: number): { label: string; color: string } {
  if (score >= 75) return { label: 'High pattern overlap', color: 'green' };
  if (score >= 50) return { label: 'Moderate pattern overlap', color: 'amber' };
  if (score >= 30) return { label: 'Some pattern overlap', color: 'orange' };
  return { label: 'Limited pattern overlap', color: 'red' };
}

// ── Helper: Call ML Service ───────────────────────────────────────────────

async function callMLService(symptoms: string[]): Promise<any> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000); // 8 second timeout
    
    const response = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.warn(`ML service unavailable: ${response.status}`);
      return null;
    }

    const data = await response.json();
    console.log('ML service response received successfully');
    return data;
  } catch (error: any) {
    if (error.name === 'AbortError') {
      console.warn('ML service call timed out after 8 seconds');
    } else {
      console.warn('ML service call failed:', error.message);
    }
    return null;
  }
}

router.post('/extract', authMiddleware, async (req: AuthRequest, res: Response) => {
  try {
    const { symptoms } = req.body;

    if (!symptoms || typeof symptoms !== 'string') {
      return res.status(400).json({ error: 'Symptoms text is required' });
    }

    const result = await llmService.extractSymptoms(symptoms);

    res.json({
      symptoms_found: result.symptoms,
      severity: result.severity,
      duration: result.duration,
    });
  } catch (error: any) {
    console.error('Extract symptoms error:', error);
    res.status(500).json({ error: error.message || 'Failed to extract symptoms' });
  }
});

router.post('/followup', authMiddleware, async (req: AuthRequest, res: Response) => {
  try {
    const { symptoms } = req.body;

    if (!symptoms || !Array.isArray(symptoms)) {
      return res.status(400).json({ error: 'Symptoms array is required' });
    }

    const questions = await llmService.generateFollowUpQuestions(symptoms);

    res.json({ questions });
  } catch (error: any) {
    console.error('Generate follow-up error:', error);
    res.status(500).json({ error: error.message || 'Failed to generate follow-up questions' });
  }
});

router.post('/diagnose', authMiddleware, async (req: AuthRequest, res: Response) => {
  try {
    const { symptoms, answers } = req.body;

    if (!symptoms || !Array.isArray(symptoms)) {
      return res.status(400).json({ error: 'Symptoms array is required' });
    }

    const followUpAnswers = answers || [];

    // ── Stage 1: Symptom Preprocessing & Gate ─────────────────────────────
    const { symptoms: processedSymptoms, gate } = preprocessSymptoms(symptoms);

    if (!gate.passed) {
      return res.status(400).json({
        error: gate.error,
        message: 'Please add at least 3 symptoms for a reliable result.',
        symptom_count: processedSymptoms.length,
        gate_passed: false,
      });
    }

    // ── Stage 2: ML Prediction ─────────────────────────────────────────────
    const mlResult = await callMLService(processedSymptoms);

    let mlEnsemble: any = null;
    let mlModelPredictions: any = null;

    if (mlResult && !mlResult.error) {
      mlEnsemble = mlResult.ensemble_prediction;
      mlModelPredictions = mlResult.model_predictions;
    }

    // Use adjusted_ml from ML service response directly (adjusted_ml variable is declared later)
    const earlyAdjustedMl: number = mlResult?.adjusted_ml ?? mlResult?.weighted_ml_raw ?? mlEnsemble?.confidence ?? 0;

    // Detect whether the new group-based models are loaded (mlEnsemble.disease is a group name)
    // vs. old individual-disease models.
    const ALL_ML_GROUPS = new Set([
      'respiratory', 'cardiovascular', 'neurological', 'psychiatric',
      'gastrointestinal', 'musculoskeletal', 'skin', 'infectious',
      'endocrine', 'renal_urinary', 'reproductive', 'hematologic',
      'oncologic', 'ophthalmic', 'ent', 'trauma_injury',
      'autoimmune', 'poisoning', 'nutritional', 'dental_oral',
    ]);
    const mlPredictsGroup = !!(mlEnsemble && ALL_ML_GROUPS.has((mlEnsemble.disease ?? '').toLowerCase()));
    const mlGroup: string | null = (mlPredictsGroup && earlyAdjustedMl >= 30)
      ? (mlEnsemble!.disease as string).toLowerCase()
      : null;

    // RAG search — filtered to ML group when the new group models are confident
    let ragResults = mlGroup
      ? ragService.searchDiseasesInGroup(processedSymptoms, mlGroup)
      : ragService.searchDiseases(processedSymptoms);

    // Fall back to full search when group filtering yields poor results —
    // this happens when ML predicted the wrong category for these symptoms.
    if (mlGroup) {
      const topGroupScore = ragResults[0]?.relevance ?? 0;
      const meaningfulHits = ragResults.filter(r => r.relevance > 0).length;
      if (meaningfulHits < 2 || topGroupScore < 30) {
        ragResults = ragService.searchDiseases(processedSymptoms);
      }
    }

    const topRagDiseases = ragResults.slice(0, 3).map(r => r.disease);

    // Build top3Diseases for RAG validation.
    // With group models: ML predicts a category, not a disease — fill all slots from RAG.
    // With legacy individual-disease models: fill ML's disease first when confident.
    const top3Diseases: string[] = [];

    if (mlEnsemble && !mlPredictsGroup && earlyAdjustedMl >= 30) {
      // Legacy individual-disease ML — add its top pick then other confident model picks
      top3Diseases.push(mlEnsemble.disease);
      if (mlModelPredictions) {
        const otherDiseases = Object.values(mlModelPredictions)
          .filter((m: any) => (m as any).confidence >= 15)
          .map((m: any) => (m as any).disease)
          .filter((d: string) => !top3Diseases.includes(d));
        top3Diseases.push(...otherDiseases.slice(0, 2));
      }
    }

    // Fill remaining slots with top RAG candidates
    for (const ragDisease of topRagDiseases) {
      if (!top3Diseases.includes(ragDisease)) {
        top3Diseases.push(ragDisease);
      }
    }
    top3Diseases.splice(3);

    // ── Stage 3: RAG Symptom Validation (top 3) ────────────────────────────
    const ragCandidates: any[] = top3Diseases.length > 0 ? ragService.validateTopDiseases(processedSymptoms, top3Diseases) : [];
    const ragValidation = ragCandidates.length > 0 ? ragCandidates[0] : null;
    const ragScore = ragValidation?.rag_score ?? 0;

    // ── Stage 2C & 4: Build ML breakdown with dynamic ensemble ──────────────
    let ml_breakdown: any = null;
    let adjusted_ml = 0;
    let model_agreement_level = 'unknown';
    let top_model = 'none';
    let disagreement_penalty_val = 0;
    let dynamic_weights: number[] = [];
    let weighted_ml_raw = 0;

    if (mlEnsemble && mlResult) {
      // Use new ML service response structure
      adjusted_ml = mlResult.adjusted_ml ?? mlResult.weighted_ml_raw ?? mlEnsemble.confidence;
      weighted_ml_raw = mlResult.weighted_ml_raw ?? mlEnsemble.confidence;
      dynamic_weights = mlResult.dynamic_weights ?? [];
      disagreement_penalty_val = mlResult.disagreement_penalty ?? 0;
      model_agreement_level = mlResult.model_agreement_level ?? 'unknown';
      top_model = mlResult.top_model ?? 'unknown';

      ml_breakdown = {
        rf_prob: mlModelPredictions?.random_forest?.confidence ?? 0,
        svm_prob: mlModelPredictions?.svm?.confidence ?? 0,
        nb_prob: mlModelPredictions?.naive_bayes?.confidence ?? 0,
        dynamic_weights: dynamic_weights,
        weighted_ml_raw: Math.round(weighted_ml_raw * 10) / 10,
        disagreement_penalty: Math.round(disagreement_penalty_val * 10) / 10,
        adjusted_ml: Math.round(adjusted_ml * 10) / 10,
        model_agreement_level,
        top_model,
      };
    }

    // ── Stage 4: Final Disease — RAG always decides ────────────────────────────
    // RAG (precision-weighted symptom matching against a structured KB) is the
    // only decision-maker for the final disease. ML contributes by:
    //   • Filtering the search space to the most likely category (Stage 3)
    //   • Adding a confidence boost when it independently agrees (Stage 4B)
    //
    // Threshold 40%: at 3 reported symptoms this means at least 2 matched.
    // Lower than this and the result is too speculative to name.
    let finalDisease: string;
    let predictionSource: 'rag_primary' | 'fallback';
    let explanation: string = '';

    if (ragValidation && ragScore >= 40) {
      finalDisease = ragValidation.disease;
      predictionSource = 'rag_primary';
    } else {
      finalDisease = 'Unknown Condition';
      predictionSource = 'fallback';
    }

    // ── Stage 4B: ML Confidence Boost ─────────────────────────────────────────
    // Group models (new): boost when ML's predicted group matches the group of
    //   RAG's final disease — confirms we're searching the right category.
    //   Random chance = 100/20 = 5% with 20 groups.
    // Legacy individual-disease models: boost when ML independently agrees on the
    //   same specific disease as RAG. Random chance = 100/773 ≈ 0.13%.
    let mlAgreementBoost = 0;
    let mlSignalMultiple = 0;
    let mlAgreesWithFinal = false;

    if (mlPredictsGroup && mlGroup && adjusted_ml >= 30) {
      // Group match boost
      const ragDiseaseGroup = ragService.getDiseaseGroup(finalDisease);
      if (ragDiseaseGroup === mlGroup) {
        mlAgreesWithFinal = true;
        // Boost scales with ML group confidence (baseline random = 5%)
        mlSignalMultiple = Math.round(adjusted_ml / (100 / 20));
        mlAgreementBoost = adjusted_ml >= 70 ? 10
          : adjusted_ml >= 50 ? 7
          : 5;
      }
    } else if (!mlPredictsGroup) {
      // Legacy individual-disease agreement boost
      const RANDOM_CHANCE_PCT = 100 / 773;
      mlAgreesWithFinal = !!(mlEnsemble &&
        mlEnsemble.disease.toLowerCase().trim() === finalDisease.toLowerCase().trim());
      if (mlAgreesWithFinal && adjusted_ml > 0) {
        mlSignalMultiple = Math.round(adjusted_ml / RANDOM_CHANCE_PCT);
        mlAgreementBoost = mlSignalMultiple >= 50 ? 15
          : mlSignalMultiple >= 20 ? 10
          : mlSignalMultiple >= 10 ? 6
          : mlSignalMultiple >= 5  ? 3
          : mlSignalMultiple >= 2  ? 1
          : 0;
        const allModelsAgree = mlModelPredictions && Object.values(mlModelPredictions)
          .every((m: any) => m.disease?.toLowerCase().trim() === finalDisease.toLowerCase().trim());
        if (allModelsAgree) mlAgreementBoost += 5;
      }
    }

    // ── Stage 4C: Final Confidence ─────────────────────────────────────────────
    // Base = RAG score (symptom match quality), plus ML boost when groups agree.
    // Capped at 95% — AI tools should never claim certainty.
    let final_confidence: number;
    if (predictionSource === 'rag_primary') {
      final_confidence = ragScore + mlAgreementBoost;
    } else {
      final_confidence = 20; // fallback: minimal signal, show as weak
    }

    final_confidence = Math.round(Math.min(95, Math.max(0, final_confidence)) * 10) / 10;

    const { label: confidence_label, color: confidence_color } = getConfidenceLabel(final_confidence);

    // Build a rich KB-based explanation as the baseline — used when Ollama is
    // unavailable or times out. Always informative regardless of LLM status.
    const kbInfo = ragService.getDiseaseInfo(finalDisease);
    function buildKBExplanation(): string {
      const matched = ragValidation?.matched ?? [];
      const missing = ragValidation?.missing ?? [];
      const description = kbInfo?.description && !kbInfo.description.startsWith('Auto-generated')
        ? kbInfo.description
        : null;

      let text = `**Closest symptom-pattern match:** ${finalDisease}\n\n`;

      if (description) {
        text += `**About this condition:**\n${description}\n\n`;
      } else {
        text += `**About this condition:**\nThe system found a symptom-pattern overlap with this condition in its reference data. This does not confirm that you have it.\n\n`;
      }

      text += `**Why it appeared in the results:**\nYou reported: *${processedSymptoms.join(', ')}*. `;

      if (matched.length > 0) {
        text += `The reference profile also includes **${matched.join(', ')}**. `;
      }

      if (missing.length > 0 && missing.length <= 5) {
        text += `Other symptoms in the reference profile include ${missing.join(', ')}. You did not report these, but that does not confirm or rule out a condition.\n\n`;
      } else {
        text += '\n\n';
      }

      text += `**Important:** The pattern score is not a probability, diagnosis, or measure of how likely you have this condition. The result cannot assess your full medical history or replace a clinician's evaluation.\n\n`;

      text += `**What to do next:**\nShare when your symptoms began, how they changed, and any existing conditions, allergies, or medicines with a qualified healthcare professional. Ask a clinician or pharmacist before starting, stopping, or changing medicines.\n\n`;

      // Methodology section for faculty/transparency
      text += `**How the system compared patterns:**\n`;
      if (predictionSource === 'rag_primary') {
        text += `The system's reference data produced a **${ragScore.toFixed(0)}% pattern score** for symptom overlap. This score is not calibrated as a probability. `;
        if (mlPredictsGroup && mlGroup) {
          if (mlAgreementBoost > 0) {
            text += `A machine-learning model also selected the broad **${mlGroup}** category. `;
          } else {
            text += `A machine-learning model suggested the broad **${mlGroup}** category, but this match came from the broader symptom search. `;
          }
        } else if (mlAgreementBoost > 0) {
          text += `A separate machine-learning model agreed with the broad pattern category. `;
        }
      } else {
        text += `Symptom patterns did not match one condition clearly. The displayed result is a weak signal, not a diagnosis.\n\n`;
      }

      text += `⚠️ **Disclaimer:** This AI-assisted screening is informational only. It does not diagnose conditions or recommend treatment. Please consult a qualified healthcare provider for personal advice.`;
      return text;
    }

    // Methodology context passed to the LLM for richer, more transparent explanations
    const llmMethodCtx = {
      predictionSource,
      ragScore: Math.round(ragScore * 10) / 10,
      adjusted_ml: Math.round(adjusted_ml * 10) / 10,
      mlGroup: mlGroup ?? undefined,
      mlAgreementBoost,
      final_confidence,
      confidence_label,
    };

    // Generate patient-friendly explanation with timeout safety
    try {
      if (predictionSource !== 'fallback') {
        if (USE_LLM && llmService.isLLMEnabled()) {
          const kbFallback = buildKBExplanation();
          explanation = await Promise.race<string>([
            llmService.generateExplanation(finalDisease, processedSymptoms, llmMethodCtx),
            new Promise(resolve => setTimeout(() => resolve(kbFallback), 6500)),
          ]);
          // If Ollama returned something too short, use the richer KB explanation
          if (explanation.trim().length < 150) {
            explanation = kbFallback;
          }
        } else {
          explanation = buildKBExplanation();
        }
      } else {
        explanation = `The symptoms you reported (${processedSymptoms.join(', ')}) did not match any single condition with sufficient confidence.\n\nCommon causes of these combined symptoms can vary widely. Please consult a healthcare provider for proper evaluation.\n\n⚠️ This is an AI-assisted analysis for informational purposes only.`;
      }
    } catch (error) {
      explanation = buildKBExplanation();
      console.error('Explanation generation error:', error);
    }

    // ── Build all predictions list ──────────────────────────────────────────
    const allPredictions: any[] = [];

    // For group-based ML: show the group prediction as context, not as a disease candidate.
    // For legacy individual-disease ML: include it as a proper candidate if confident.
    if (mlEnsemble && adjusted_ml >= 30) {
      if (mlPredictsGroup) {
        allPredictions.push({
          disease: `[ML category: ${mlEnsemble.disease}]`,
          confidence: Math.round(adjusted_ml * 10) / 10,
          source: 'ml_group_filter',
          used_in_final: false,
        });
      } else {
        allPredictions.push({
          disease: mlEnsemble.disease,
          confidence: Math.round(adjusted_ml * 10) / 10,
          source: 'ml_ensemble',
          used_in_final: mlEnsemble.disease === finalDisease,
        });
      }
    }

    for (const cand of ragCandidates) {
      if (cand.rag_score > 0 && !allPredictions.some(p => p.disease === cand.disease)) {
        allPredictions.push({
          disease: cand.disease,
          confidence: Math.round(cand.rag_score * 10) / 10,
          source: 'rag_validation',
          used_in_final: cand.disease === finalDisease,
        });
      }
    }

    // ── Build RAG breakdown ─────────────────────────────────────────────────
    const rag_breakdown = ragValidation
      ? {
          rag_score: ragScore,
          score: ragScore,
          matched: ragValidation.matched,
          missing: ragValidation.missing,
          matchedSymptoms: ragValidation.matched,
          missingSymptoms: ragValidation.missing,
          red_flags_present: ragValidation.red_flags_present,
          red_flags_missing: ragValidation.red_flags_missing,
        }
      : null;

    // ── Build ML breakdown additions ───────────────────────────────────────
    if (!ml_breakdown) {
      ml_breakdown = {
        rf_prob: 0, svm_prob: 0, nb_prob: 0,
        dynamic_weights: [],
        weighted_ml_raw: 0,
        disagreement_penalty: 0,
        adjusted_ml: 0,
        model_agreement_level: 'unavailable',
        top_model: 'none',
      };
    }
    // Attach ML signal info to the breakdown
    ml_breakdown.predicts_group = mlPredictsGroup;
    ml_breakdown.ml_group = mlGroup;
    ml_breakdown.agrees_with_rag = mlAgreesWithFinal;
    ml_breakdown.signal_multiple = mlSignalMultiple;
    ml_breakdown.agreement_boost = mlAgreementBoost;

    // ── Build Methodology Explanation ───────────────────────────────────────
    // Explain system pattern scores without presenting them as probabilities.
    const methodsUsed: string[] = ['rag'];
    const primaryMethod = 'rag';
    let methodologyExplanation = '';

    if (predictionSource === 'rag_primary') {
      if (mlPredictsGroup && mlGroup && mlAgreementBoost > 0) {
        methodologyExplanation = `The symptom pattern search selected ${finalDisease} as its closest reference match (${ragScore.toFixed(1)}% pattern score). A separate model suggested the broad "${mlGroup}" category. The displayed score is not a probability or diagnosis.`;
      } else if (mlPredictsGroup && mlGroup) {
        methodologyExplanation = `The symptom pattern search selected ${finalDisease} as its closest reference match (${ragScore.toFixed(1)}% pattern score). A separate model suggested the "${mlGroup}" category, so the system compared against the broader reference set. The displayed score is not a probability or diagnosis.`;
      } else if (mlAgreementBoost > 0) {
        methodologyExplanation = `The symptom pattern search selected ${finalDisease} as its closest reference match (${ragScore.toFixed(1)}% pattern score), and a separate model agreed with the pattern. The displayed score is not a probability or diagnosis.`;
      } else {
        const mlStatus = mlEnsemble ? 'A separate model produced a different pattern.' : 'The separate machine-learning service was unavailable.';
        methodologyExplanation = `The symptom pattern search selected ${finalDisease} as its closest reference match (${ragScore.toFixed(1)}% pattern score). ${mlStatus} The displayed score is not a probability or diagnosis.`;
      }
    } else {
      methodologyExplanation = `The reported symptoms did not produce a clear match in the reference data (best pattern score: ${ragScore.toFixed(1)}%). This is not a diagnosis. Consider discussing your symptoms with a healthcare professional.`;
    }

    // ── Safety ───────────────────────────────────────────────────────────────
    const safety = {
      disclaimer: 'This is not a medical diagnosis. Consult a doctor.',
      recommend_doctor: final_confidence < 70,
      symptom_count: processedSymptoms.length,
      gate_passed: true,
    };

    // ── Final Response ───────────────────────────────────────────────────────
    const finalPrediction = {
      top_result: {
        disease: finalDisease,
        final_confidence,
        confidence_label,
        confidence_color,
      },
      ml_breakdown,
      rag_breakdown,
      safety,
      candidates: [], // ragCandidates.map(c => ({
      //   disease: c.disease,
      //   rag_score: c.rag_score,
      //   matched: c.matched,
      //   missing: c.missing,
      // })),
      // Backwards-compatible legacy fields
      disease: finalDisease,
      confidence: final_confidence,
      all_predictions: allPredictions,
      explanation,
      explanation_pending: false,
      symptoms_found: processedSymptoms,
      prediction_source: predictionSource,
      ml_available: !!mlEnsemble,
      ml_confidence: Math.round(adjusted_ml * 10) / 10,
      ml_group: mlPredictsGroup ? mlEnsemble?.disease ?? null : null,
      ml_disease: mlPredictsGroup ? null : (mlEnsemble?.disease ?? null),
      ml_used: mlAgreementBoost > 0,
      symptom_match_score: ragScore,
      rag_validation: ragValidation,
      rag_context_preview: ragValidation
        ? `RAG validated ${ragValidation.disease} with ${ragScore.toFixed(1)}% symptom alignment.`
        : 'No RAG validation could be performed for the top predicted diseases.',
      methodology: {
        primary_method: primaryMethod,
        methods_used: methodsUsed,
        ml_models_used: ml_breakdown,
        explanation: methodologyExplanation,
        decision_path: [],
      },
    };

    res.json(finalPrediction);
  } catch (error: any) {
    console.error('Diagnosis error:', error);
    res.status(500).json({ error: error.message || 'Failed to generate diagnosis' });
  }
});

router.get('/diseases', async (req: AuthRequest, res: Response) => {
  try {
    const diseases = ragService.getAllDiseases();
    res.json({ diseases });
  } catch (error: any) {
    console.error('Get diseases error:', error);
    res.status(500).json({ error: 'Failed to fetch diseases' });
  }
});

router.get('/disease/:name', async (req: AuthRequest, res: Response) => {
  try {
    const { name } = req.params;
    const info = ragService.getDiseaseInfo(name);
    
    if (!info) {
      return res.status(404).json({ error: 'Disease not found' });
    }
    
    res.json(info);
  } catch (error: any) {
    console.error('Get disease info error:', error);
    res.status(500).json({ error: 'Failed to fetch disease info' });
  }
});

router.get('/health', async (req: AuthRequest, res: Response) => {
  try {
    const ollamaHealth = await llmService.healthCheck();
    res.json({
      status: 'ok',
      ollama: ollamaHealth ? 'connected' : 'disconnected',
    });
  } catch (error: any) {
    res.json({
      status: 'ok',
      ollama: 'disconnected',
    });
  }
});

export default router;