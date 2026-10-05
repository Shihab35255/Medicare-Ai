import medicalKnowledge from '../data/medical-knowledge.json';
import * as datasetService from './dataset.service.js';

interface Disease {
  disease: string;
  symptoms: string[];
  description: string;
  treatments: string[];
  severity: string;
  core_symptoms?: string[];
  red_flag_symptoms?: string[];
}

interface SearchResult {
  disease: string;
  relevance: number;
  matchedSymptoms: string[];
  description: string;
  treatments: string[];
  severity: string;
  datasetPrevalence?: number; // % from CSV dataset
  datasetRecords?: number;
}

function normalizeText(text: string): string {
  // Apply synonyms first
  const lower = text.toLowerCase();
  for (const [synonym, canonicals] of Object.entries(SYMPTOM_SYNONYMS)) {
    if (lower.includes(synonym)) {
      return canonicals[0]; // Use first canonical form
    }
  }
  return lower.replace(/[^a-z0-9\s]/g, '').replace(/\s+/g, ' ');
}

function jaroWinklerSimilarity(s1: string, s2: string): number {
  // Jaro-Winkler distance implementation (0-1 score)
  const len1 = s1.length;
  const len2 = s2.length;
  if (len1 === 0 && len2 === 0) return 1;
  if (len1 === 0 || len2 === 0) return 0;

  const maxDist = Math.floor(Math.max(len1, len2) / 2) - 1;
  let matchCount = 0;
  const matches1 = new Array(len1).fill(false);
  const matches2 = new Array(len2).fill(false);

  // Find matches
  for (let i = 0; i < len1; i++) {
    const start = Math.max(0, i - maxDist);
    const end = Math.min(len2, i + maxDist + 1);
    for (let j = start; j < end; j++) {
      if (!matches2[j] && s1[i] === s2[j]) {
        matches1[i] = true;
        matches2[j] = true;
        matchCount++;
        break;
      }
    }
  }

  if (matchCount === 0) return 0;

  // Transpositions
  let transCount = 0;
  for (let i = 0; i < len1; i++) {
    if (matches1[i]) {
      while (!matches2[i]) i++;
    }
  }
  transCount /= 2;

  const jaro = ((matchCount / len1) + (matchCount / len2) + ((matchCount - transCount) / matchCount)) / 3.0;

  // Winkler prefix boost
  const prefix = Math.min(4, Math.min(len1, len2));
  let prefixMatches = 0;
  for (let i = 0; i < prefix; i++) {
    if (s1[i] === s2[i]) prefixMatches++;
    else break;
  }
  const scaling = 0.1;
  return jaro + scaling * prefixMatches * (1 - jaro);
}

/**
 * Maps ML model disease names to medical knowledge base disease names.
 * This bridges the gap between the ML training data and the KB.
 */
const DISEASE_ALIASES: Record<string, string[]> = {
  'acute bronchitis': ['Bronchitis'],
  'cystitis': ['Urinary Tract Infection', 'Bladder Disorder'],
  'conjunctivitis due to allergy': ['Allergic Rhinitis', 'Seasonal Allergies (Hay Fever)'],
  'esophagitis': ['GERD (Acid Reflux)'],
  'gastrointestinal hemorrhage': ['Gastroenteritis (Stomach Flu)'],
  'infectious gastroenteritis': ['Gastroenteritis (Stomach Flu)', 'Infectious Gastroenteritis'],
  'pneumonia': ['Pneumonia'],
  'hypoglycemia': ['Diabetes Type 1', 'Diabetes Type 2'],
  'gout': ['Arthritis', 'Gout'],
  'arthritis of the hip': ['Arthritis'],
  'bursitis': ['Arthritis', 'Bursitis'],
  'spondylosis': ['Arthritis', 'Spondylosis'],
  'diverticulitis': ['Diverticulosis'],
  // Removed 'liver disease': ['Heart Disease'] - incorrect mapping
  'nose disorder': ['Common Cold', 'Nose Disorder'],
  'fungal infection of the hair': ['Fungal Infection Of The Skin'],
  'marijuana abuse': ['Anxiety Disorder', 'Depression', 'Drug Abuse (Marijuana)'],
  'peripheral nerve disorder': ['Peripheral Nerve Disorder'],
  'complex regional pain syndrome': ['Complex Regional Pain Syndrome'],
  'spontaneous abortion': ['Pain during pregnancy', 'Problem During Pregnancy'],
  'vaginal cyst': ['Vaginal Cyst'],
  'vulvodynia': ['Vulvodynia'],
  'sprain or strain': ['Injury to the arm'],
  'injury to the arm': ['Injury to the arm'],
  'strep throat': ['Pharyngitis', 'Strep Throat'],
};

const SYMPTOM_SYNONYMS: Record<string, string[]> = {
  'head hurts': ['headache'],
  'headache or migraine': ['headache'],
  'stomach ache': ['abdominal pain', 'sharp abdominal pain'],
  'tummy pain': ['abdominal pain'],
  'chest hurts': ['chest pain', 'sharp chest pain'],
  'short of breath': ['shortness of breath'],
  'hard to breathe': ['difficulty breathing', 'shortness of breath'],
  'nauseous': ['nausea'],
  'throwing up': ['vomiting'],
  'hurts to pee': ['painful urination'],
  'frequent peeing': ['frequent urination'],
  'nervous': ['anxiety and nervousness'],
  'sad': ['depression'],
  'insomniac': ['insomnia'],
  'heart racing': ['palpitations', 'increased heart rate'],
  'dizzy': ['dizziness'],
  'stiff neck': ['neck stiffness or tightness'],
  'feverish': ['fever'],
  'rashes': ['skin rash'],
  'itching': ['itching of skin'],
  'joint hurts': ['joint pain'],
  'back hurts': ['back pain'],
  'leg pain or cramps': ['leg pain', 'leg cramps or spasms'],
  'arm hurts': ['arm pain'],
  'tired all time': ['fatigue'],
  'losing weight': ['recent weight loss'],
  'thirsty a lot': ['thirst'],
  'pee a lot': ['frequent urination', 'excessive urination at night'],
  'blurry vision': ['diminished vision', 'spots or clouds in vision'],
  'coughing': ['cough'],
  'runny nose': ['nasal congestion', 'coryza'],
  'sore throat pain': ['sore throat'],
  'ear hurts': ['ear pain'],
  'skin bumps': ['skin lesion', 'skin growth'],
  'belly bloated': ['stomach bloating', 'abdominal distention'],
  'diarrhea loose stool': ['diarrhea'],
  'constipated': ['constipation'],
  'burning pee': ['painful urination'],
  'blood in stool': ['blood in stool', 'melena', 'rectal bleeding'],
  'vomit blood': ['vomiting blood'],
  'yellow skin': ['jaundice'],
  'swollen legs': ['leg swelling', 'peripheral edema'],
  'numb tingling': ['loss of sensation', 'paresthesia'],
  'weak muscles': ['weakness'],
  'confused': ['disturbance of memory', 'depressive or psychotic symptoms'],
  'seeing spots': ['spots or clouds in vision'],
  'ringing ears': ['ringing in ear'],
  'trouble swallowing': ['difficulty in swallowing'],
  'lump throat': ['lump in throat'],
  'hoarse voice change': ['hoarse voice'],
  'night sweats': ['sweating'],
  'unexplained bruises': ['abnormal appearing skin'],
  'easy bleeding': ['nosebleed', 'bleeding gums'],
  'dark urine': ['blood in urine'],
  'pale skin': ['abnormal appearing skin'],
  'big lymph nodes': ['swollen lymph nodes'],
};

/**
 * Look up a disease in the knowledge base, trying exact match first,
 * then aliases if no exact match is found.
 */
function findDiseaseInKB(diseaseName: string): Disease | undefined {
  const diseases = medicalKnowledge as Disease[];
  const normalizedTarget = normalizeText(diseaseName);

  // Pass 1: exact match only — prevents "cystitis" → "cholecystitis" substring collision
  const exact = diseases.find(d => normalizeText(d.disease) === normalizedTarget);
  if (exact) return exact;

  // Pass 2: aliases (exact match against alias names)
  const aliasKey = diseaseName.toLowerCase();
  const aliases = DISEASE_ALIASES[aliasKey] ?? DISEASE_ALIASES[diseaseName];
  if (aliases) {
    for (const alias of aliases) {
      const normalizedAlias = normalizeText(alias);
      const aliasMatch = diseases.find(d => normalizeText(d.disease) === normalizedAlias);
      if (aliasMatch) return aliasMatch;
    }
  }

  // Pass 3: query-contains-disease partial match (safe direction)
  // e.g. query "flu" finds "Flu" even without exact casing; never matches disease names
  // that are LONGER than the query (no "cystitis" → "cholecystitis" false matches)
  if (normalizedTarget.length >= 4) {
    const partial = diseases.find(d => {
      const norm = normalizeText(d.disease);
      return norm !== normalizedTarget && normalizedTarget.includes(norm) && norm.length >= 4;
    });
    if (partial) return partial;
  }

  return undefined;
}

function calculateSimilarity(symptoms: string[], diseaseSymptoms: string[]): number {
  if (symptoms.length === 0 || diseaseSymptoms.length === 0) return 0;

  const normalizedInput = symptoms.map(normalizeText);
  const normalizedDb = diseaseSymptoms.map(normalizeText);

  let matchCount = 0;
  for (const inputSymptom of normalizedInput) {
    if (normalizedDb.some(dbSymptom => dbSymptom.includes(inputSymptom) || inputSymptom.includes(dbSymptom))) {
      matchCount++;
    }
  }

  // Precision: how many of the user's symptoms does this disease explain?
  // Recall: how much of the disease profile is covered by user's symptoms?
  // Weight precision 70% — user's symptoms are the primary signal.
  const precision = matchCount / symptoms.length;
  const recall = matchCount / diseaseSymptoms.length;
  return (0.7 * precision + 0.3 * recall) * 100;
}

export function searchDiseases(symptoms: string[]): SearchResult[] {
  const diseases = medicalKnowledge as Disease[];
  
  const results: SearchResult[] = diseases.map(disease => {
    const relevance = calculateSimilarity(symptoms, disease.symptoms);
    const normalizedInput = symptoms.map(normalizeText);
    const matched = disease.symptoms.filter(ds => 
      normalizedInput.some(is => is.includes(normalizeText(ds)) || normalizeText(ds).includes(is))
    );

    // Get dataset profile for this disease using the service lookup
    const dsProfile = datasetService.getDiseaseProfile(disease.disease);
    
    return {
      disease: disease.disease,
      relevance,
      matchedSymptoms: matched,
      description: disease.description,
      treatments: disease.treatments,
      severity: disease.severity,
      datasetRecords: dsProfile?.totalRecords,
    };
  });
  
  return results
    .filter(r => r.relevance >= 0)
    .sort((a, b) => b.relevance - a.relevance)
    .slice(0, 10);
}

export function getDiseaseInfo(diseaseName: string): SearchResult | null {
  const diseases = medicalKnowledge as Disease[];
  const normalized = normalizeText(diseaseName);
  
  const disease = diseases.find(d => normalizeText(d.disease).includes(normalized));
  
  if (!disease) return null;
  
  return {
    disease: disease.disease,
    relevance: 100,
    matchedSymptoms: disease.symptoms,
    description: disease.description,
    treatments: disease.treatments,
    severity: disease.severity,
  };
}

export function getAllDiseases(): string[] {
  const diseases = medicalKnowledge as Disease[];
  return diseases.map(d => d.disease);
}



export interface SymptomValidationResult {
  disease: string;
  matchedSymptoms: string[];
  missingSymptoms: string[];
  rag_score: number;
  red_flags_present: string[];
  red_flags_missing: string[];
}

/**
 * Stage 3: RAG Symptom Validation with weighted symptom matching
 * Core symptoms (must-have) weight = 1.5, Supporting symptoms weight = 1.0
 * Red flag penalties/rewards applied
 */
export function validateSymptomMatches(
  symptoms: string[],
  diseaseName: string
): SymptomValidationResult {
  const disease = findDiseaseInKB(diseaseName);
  if (!disease) {
    return {
      disease: diseaseName,
      matchedSymptoms: [],
      missingSymptoms: [],
      rag_score: 0,
      red_flags_present: [],
      red_flags_missing: [],
    };
  }

  const normalizedInput = symptoms.map(normalizeText);
  const normalizedDiseaseSymptoms = disease.symptoms.map(normalizeText);

  const coreSymptoms = (disease.core_symptoms || []).map(normalizeText);
  const redFlagSymptoms = (disease.red_flag_symptoms || []).map(normalizeText);

  const matchedSymptoms: string[] = [];
  const missingSymptoms: string[] = [];
  let weightedMatch = 0;   // sum of weights for matched disease symptoms (recall side)
  let weightedTotal = 0;   // sum of all disease symptom weights
  let inputMatched = 0;    // count of user's symptoms that were matched (precision side)

  for (let i = 0; i < disease.symptoms.length; i += 1) {
    const dbSymptom = disease.symptoms[i];
    const normalizedDbSymptom = normalizedDiseaseSymptoms[i];

    // Determine weight: core = 1.5, supporting = 1.0
    const isCore = coreSymptoms.some(cs => cs === normalizedDbSymptom);
    const weight = isCore ? 1.5 : 1.0;
    weightedTotal += weight;

    const matched = normalizedInput.some(inputSymptom =>
      normalizedDbSymptom.includes(inputSymptom) || inputSymptom.includes(normalizedDbSymptom)
    );

    if (matched) {
      matchedSymptoms.push(dbSymptom);
      weightedMatch += weight;
    } else {
      missingSymptoms.push(dbSymptom);
    }
  }

  // Also count how many of the user's input symptoms were explained by this disease
  for (const inputSymptom of normalizedInput) {
    if (normalizedDiseaseSymptoms.some(ds => ds.includes(inputSymptom) || inputSymptom.includes(ds))) {
      inputMatched++;
    }
  }

  const recallScore = weightedTotal === 0 ? 0 : weightedMatch / weightedTotal;
  const precisionScore = normalizedInput.length === 0 ? 0 : inputMatched / normalizedInput.length;
  // Weight precision 70%: "how much of what the user reported does this disease explain?"
  let ragScore = (0.7 * precisionScore + 0.3 * recallScore) * 100;

  // Red flag checks
  const redFlagsPresent: string[] = [];
  const redFlagsMissing: string[] = [];

  for (const redFlag of redFlagSymptoms) {
    const isPresent = normalizedInput.some(is => is.includes(redFlag) || redFlag.includes(is));
    if (isPresent) {
      redFlagsPresent.push(redFlag);
    } else {
      redFlagsMissing.push(redFlag);
    }
  }

  // Boost score only when red flags ARE present (confirms severe case).
  // Missing red flags just means a mild presentation — no penalty.
  if (redFlagsPresent.length > 0 && redFlagsMissing.length === 0) {
    ragScore = Math.min(100, ragScore * 1.15);
  }

  return {
    disease: disease.disease,
    matchedSymptoms,
    missingSymptoms,
    rag_score: Math.round(ragScore * 10) / 10,
    red_flags_present: redFlagsPresent,
    red_flags_missing: redFlagsMissing,
  };
}

/**
 * Get RAG validation scores for top 3 predicted diseases
 */
export function validateTopDiseases(
  symptoms: string[],
  topDiseases: string[]
): Array<{
  disease: string;
  rag_score: number;
  matched: string[];
  missing: string[];
  red_flags_present: string[];
  red_flags_missing: string[];
}> {
  return topDiseases
    .slice(0, 3)
    .map(diseaseName => {
      const validation = validateSymptomMatches(symptoms, diseaseName);
      return {
        disease: validation.disease,
        rag_score: validation.rag_score,
        matched: validation.matchedSymptoms,
        missing: validation.missingSymptoms,
        red_flags_present: validation.red_flags_present,
        red_flags_missing: validation.red_flags_missing,
      };
    })
    .sort((a, b) => b.rag_score - a.rag_score);
}

// ── Disease Group Classification ─────────────────────────────────────────────
// Mirrors GROUP_KEYWORD_RULES and DISEASE_GROUP_OVERRIDES from train_models.py
// so the backend can map any KB disease to its 20-group category.

const GROUP_KEYWORD_RULES_TS: Array<[string, string[]]> = [
  ['respiratory',     ['lung', 'bronch', 'pulmon', 'pneum', 'asthma', 'copd',
                       'emphys', 'pleural', 'trachei', 'ards', 'whooping',
                       'tuberculosis', 'covid', 'respiratory']],
  ['cardiovascular',  ['heart', 'cardiac', 'cardio', 'coronar', 'atrial',
                       'ventricular', 'arrhythmia', 'angina', 'aortic', 'aorta',
                       'arterial', 'myocarditis', 'endocarditis', 'pericarditis',
                       'thrombophlebitis', 'atherosclerosis']],
  ['neurological',    ['brain', 'cerebr', 'neural', 'neuro', 'epilep', 'migraine',
                       'stroke', 'meningit', 'encephalit', 'parkinson', 'alzheimer',
                       'dementia', 'sclerosis', 'palsy', 'seizure', 'headache',
                       'intracranial', 'subarachnoid', 'subdural', 'epidural',
                       'hydrocephalus', 'ataxia', 'narcolepsy', 'myasthenia',
                       'guillain', 'neuralgia', 'neuropath', 'mononeuritis']],
  ['psychiatric',     ['depress', 'anxiety', 'bipolar', 'schizo', 'psychot',
                       'panic', 'ptsd', 'ocd', 'phobia', 'personality disorder',
                       'dissociati', 'somatiz', 'factitious', 'dysthymic',
                       'drug abuse', 'drug withdrawal', 'alcohol abuse',
                       'alcohol intox', 'alcohol withdrawal', 'marijuana',
                       'substance', 'eating disorder', 'adhd', 'autism',
                       'asperger', 'conduct disorder', 'oppositional',
                       'impulse control', 'adjustment reaction', 'acute stress',
                       'postpartum depression', 'neurosis']],
  ['gastrointestinal', ['gastro', 'intestin', 'bowel', 'colon', 'rectal',
                        'liver', 'hepat', 'pancrea', 'esophag', 'stomach',
                        'append', 'diverticul', 'hernia', 'peritonit',
                        'cholecyst', 'choledoch', 'gallstone', 'cirrhosis',
                        'volvulus', 'ileus', 'indigestion', 'hemorrhoid',
                        'anal fiss', 'anal fistul', 'perirectal']],
  ['musculoskeletal', ['bone', 'joint', 'muscl', 'tendin', 'ligament',
                       'fracture', 'dislocation', 'arthrit', 'spondyl',
                       'osteo', 'bursit', 'lumbago', 'sciatica', 'disc',
                       'spinal stenosis', 'scoliosis', 'fibromyalg',
                       'carpal tunnel', 'epicondyl', 'rotator', 'plantar',
                       'sprain', 'strain', 'torticoll', 'gout', 'myosit']],
  ['skin',            ['skin', 'dermat', 'acne', 'eczema', 'psoriasis',
                       'rosacea', 'melanoma', 'alopecia', 'fungal infection',
                       'impetigo', 'cellulitis', 'scabies', 'lice',
                       'chickenpox', 'shingles', 'herpes zoster', 'wart',
                       'molluscum', 'pemphigus', 'lichen', 'seborrheic',
                       'vitiligo', 'pityriasis', 'onychomycosis', 'athlete']],
  ['infectious',      ['sepsis', 'malaria', 'typhoid', 'dengue', 'lyme',
                       'rocky mountain', 'mononucleosis', 'mumps', 'scarlet',
                       'toxoplasmos', 'histoplasmos', 'cryptococcosis',
                       'aspergillosis', 'cysticercosis', 'trichinosis',
                       'parasitic', 'pinworm', 'sporotrichosis', 'syphilis',
                       'gonorrhea', 'chlamydia', 'trichomonas', 'granuloma',
                       'lymphogranuloma', 'hiv', 'cat scratch', 'valley fever']],
  ['endocrine',       ['diabet', 'thyroid', 'insulin', 'hypoglycemi',
                       'hyperglycemi', 'adrenal', 'pituitary', 'parathyroid',
                       'graves', 'hashimoto', 'goiter', 'cushing',
                       'hypercalcemia', 'hypocalcemia', 'hyponatremia',
                       'hypernatremia', 'hypokalemia', 'hyperkalemia',
                       'hyperlipidemia', 'metabolic disorder', 'hormone disorder',
                       'hyperosmotic', 'siadh', 'glucocorticoid']],
  ['renal_urinary',   ['kidney', 'renal', 'urinar', 'bladder', 'urethral',
                       'cystitis', 'pyelonephrit', 'hydronephrosis',
                       'nephrosis', 'ureterl', 'urethritis', 'incontinence',
                       'vesicoureteral']],
  ['reproductive',    ['pregnan', 'obstetric', 'uterine', 'ovarian', 'cervical',
                       'endometri', 'menstrual', 'menopause', 'vaginal',
                       'vulvar', 'vaginit', 'prostate', 'testicular',
                       'epididym', 'erectile', 'priapism', 'varicocele',
                       'hydrocele', 'mastitis', 'polycystic ovarian', 'pelvic',
                       'placenta', 'preeclampsia', 'abortion', 'ectopic']],
  ['hematologic',     ['anemia', 'hemophilia', 'leukemia', 'lymphoma',
                       'thrombocytopenia', 'polycythemia', 'thalassemia',
                       'sickle cell', 'coagulation', 'bleeding disorder',
                       'myelodysplastic', 'lymphedema', 'hypovolemia',
                       'rhabdomyolysis', 'white blood cell', 'von willebrand',
                       'fluid overload', 'spherocytosis', 'g6pd']],
  ['oncologic',       ['cancer', 'tumor', 'malignant', 'sarcoma', 'carcinoid',
                       'metastatic', 'myeloma', 'ependymoma', 'meningioma']],
  ['ophthalmic',      ['eye', 'ocular', 'optic', 'retinal', 'cornea', 'corneal',
                       'conjunctivit', 'glaucoma', 'cataract', 'macular',
                       'blephar', 'stye', 'chalazion', 'iridocyclitis',
                       'uveitis', 'scleritis', 'endophthalmitis', 'vitreous',
                       'myopia', 'hyperopia', 'astigmatism', 'presbyopia',
                       'amblyopia', 'pterygium', 'pinguecula', 'floaters']],
  ['ent',             ['ear', 'otitis', 'hearing', 'tinnitus', 'sinus',
                       'nasal', 'pharyngit', 'tonsil', 'laryngit', 'throat',
                       'mastoid', 'cholesteatoma', 'labyrinthitis', 'meniere',
                       'vertigo', 'bppv', 'vocal cord', 'nose disorder',
                       'salivary', 'eustachian', 'otosclerosis']],
  ['trauma_injury',   ['open wound', 'injury', 'crush', 'burn', 'frostbite',
                       'heat stroke', 'heat exhaustion', 'hypothermia',
                       'envenomation', 'insect bite', 'carbon monoxide',
                       'birth trauma']],
  ['autoimmune',      ['lupus', 'scleroderma', 'sjogren', 'vasculitis',
                       'rheumatic fever', 'primary immunodeficiency',
                       'hypergammaglobulinemia', 'amyloidosis']],
  ['poisoning',       ['poison', 'overdose', 'intoxication', 'drug reaction',
                       'drug poisoning', 'toxin']],
  ['nutritional',     ['deficiency', 'scurvy', 'malnutrition',
                       'protein deficiency', 'folate', 'vitamin']],
  ['dental_oral',     ['dental', 'tooth', 'teeth', 'gum disease', 'oral',
                       'jaw disorder', 'aphthous', 'temporomandibular']],
];

const DISEASE_GROUP_OVERRIDES_KB: Record<string, string> = {
  'flu': 'respiratory',
  'influenza': 'respiratory',
  'common cold': 'respiratory',
  'covid-19': 'respiratory',
  'croup': 'ent',
  'sarcoidosis': 'respiratory',
  'empyema': 'respiratory',
  'atelectasis': 'respiratory',
  'pleural effusion': 'respiratory',
  'high blood pressure': 'cardiovascular',
  'deep vein thrombosis (dvt)': 'cardiovascular',
  'varicose veins': 'cardiovascular',
  'venous insufficiency': 'cardiovascular',
  'raynaud disease': 'cardiovascular',
  'sick sinus syndrome': 'cardiovascular',
  'orthostatic hypotension': 'cardiovascular',
  'hypertensive heart disease': 'cardiovascular',
  'peripheral arterial disease': 'cardiovascular',
  'peripheral arterial embolism': 'cardiovascular',
  'tension headache': 'neurological',
  'bell palsy': 'neurological',
  'transient ischemic attack': 'neurological',
  'essential tremor': 'neurological',
  'cranial nerve palsy': 'neurological',
  'hemiplegia': 'neurological',
  'pseudotumor cerebri': 'neurological',
  'syringomyelia': 'neurological',
  'wernicke korsakoff syndrome': 'neurological',
  'moyamoya disease': 'neurological',
  'complex regional pain syndrome': 'neurological',
  'tourette syndrome': 'neurological',
  'tic (movement) disorder': 'neurological',
  'autonomic nervous system disorder': 'neurological',
  'brachial neuritis': 'neurological',
  'drug withdrawal': 'psychiatric',
  'neurosis': 'psychiatric',
  'adjustment reaction': 'psychiatric',
  'acute stress reaction': 'psychiatric',
  'conversion disorder': 'psychiatric',
  'infectious gastroenteritis': 'gastrointestinal',
  'noninfectious gastroenteritis': 'gastrointestinal',
  'gallstone': 'gastrointestinal',
  'indigestion': 'gastrointestinal',
  'hemorrhoids': 'gastrointestinal',
  'dumping syndrome': 'gastrointestinal',
  'meckel diverticulum': 'gastrointestinal',
  'pyloric stenosis': 'gastrointestinal',
  'zenker diverticulum': 'gastrointestinal',
  'volvulus': 'gastrointestinal',
  'ileus': 'gastrointestinal',
  'peritonitis': 'gastrointestinal',
  'ischemia of the bowel': 'gastrointestinal',
  'intussusception': 'gastrointestinal',
  'perirectal infection': 'gastrointestinal',
  'hirschsprung disease': 'gastrointestinal',
  'ascending cholangitis': 'gastrointestinal',
  'gastrointestinal hemorrhage': 'gastrointestinal',
  'gout': 'musculoskeletal',
  'fibromyalgia': 'musculoskeletal',
  'lumbago': 'musculoskeletal',
  'sciatica': 'musculoskeletal',
  'scoliosis': 'musculoskeletal',
  'torticollis': 'musculoskeletal',
  'bursitis': 'musculoskeletal',
  'sprain or strain': 'musculoskeletal',
  'plantar fasciitis': 'musculoskeletal',
  'muscle spasm': 'musculoskeletal',
  'restless leg syndrome': 'musculoskeletal',
  'polymyalgia rheumatica': 'musculoskeletal',
  'reactive arthritis': 'musculoskeletal',
  'septic arthritis': 'musculoskeletal',
  'juvenile rheumatoid arthritis': 'musculoskeletal',
  'chickenpox': 'skin',
  'shingles': 'skin',
  'shingles (herpes zoster)': 'skin',
  'cold sore': 'skin',
  'impetigo': 'skin',
  'viral exanthem': 'skin',
  'viral warts': 'skin',
  'rosacea': 'skin',
  'pityriasis rosea': 'skin',
  'hidradenitis suppurativa': 'skin',
  'dyshidrosis': 'skin',
  'necrotizing fasciitis': 'skin',
  'sepsis': 'infectious',
  'malaria': 'infectious',
  'typhoid fever': 'infectious',
  'dengue fever': 'infectious',
  'lyme disease': 'infectious',
  'mononucleosis': 'infectious',
  'mumps': 'infectious',
  'scarlet fever': 'infectious',
  'herpangina': 'infectious',
  'cat scratch disease': 'infectious',
  'parasitic disease': 'infectious',
  'pinworm infection': 'infectious',
  'hpv': 'infectious',
  'human immunodeficiency virus infection (hiv)': 'infectious',
  'genital herpes': 'infectious',
  'rocky mountain spotted fever': 'infectious',
  'hypothyroidism': 'endocrine',
  'hyperthyroidism': 'endocrine',
  'goiter': 'endocrine',
  'cushing syndrome': 'endocrine',
  'obesity': 'endocrine',
  'hyperlipidemia': 'endocrine',
  'hypercholesterolemia': 'endocrine',
  'urinary tract infection': 'renal_urinary',
  'kidney stone': 'renal_urinary',
  'hydronephrosis': 'renal_urinary',
  'vesicoureteral reflux': 'renal_urinary',
  'atonic bladder': 'renal_urinary',
  'pregnancy': 'reproductive',
  'menopause': 'reproductive',
  'polycystic ovarian syndrome (pcos)': 'reproductive',
  'premature ovarian failure': 'reproductive',
  'vulvodynia': 'reproductive',
  'priapism': 'reproductive',
  'balanitis': 'reproductive',
  'varicocele of the testicles': 'reproductive',
  'hydrocele of the testicle': 'reproductive',
  'sickle cell anemia': 'hematologic',
  'sickle cell crisis': 'hematologic',
  'aplastic anemia': 'hematologic',
  'hemolytic anemia': 'hematologic',
  'iron deficiency anemia': 'hematologic',
  'anemia of chronic disease': 'hematologic',
  'thalassemia': 'hematologic',
  'lymphedema': 'hematologic',
  'hypovolemia': 'hematologic',
  'lung cancer': 'oncologic',
  'breast cancer': 'oncologic',
  'colorectal cancer': 'oncologic',
  'stomach cancer': 'oncologic',
  'liver cancer': 'oncologic',
  'pancreatic cancer': 'oncologic',
  'cervical cancer': 'oncologic',
  'ovarian cancer': 'oncologic',
  'prostate cancer': 'oncologic',
  'brain cancer': 'oncologic',
  'skin cancer': 'oncologic',
  'dry eye of unknown cause': 'ophthalmic',
  'floaters': 'ophthalmic',
  'orbital cellulitis': 'ophthalmic',
  'eye alignment disorder': 'ophthalmic',
  'meniere disease': 'ent',
  'tinnitus of unknown cause': 'ent',
  'allergy': 'ent',
  'allergy to animals': 'ent',
  'seasonal allergies (hay fever)': 'ent',
  'food allergy': 'ent',
  'strep throat': 'ent',
  'deviated nasal septum': 'ent',
  'nasal polyp': 'ent',
  'nose disorder': 'ent',
  'cholesteatoma': 'ent',
  'ear wax impaction': 'ent',
  'ear drum damage': 'ent',
  'salivary gland disorder': 'ent',
  'vocal cord polyp': 'ent',
  'tracheitis': 'ent',
  'peritonsillar abscess': 'ent',
  'open wound due to trauma': 'trauma_injury',
  'crushing injury': 'trauma_injury',
  'birth trauma': 'trauma_injury',
  'insect bite': 'trauma_injury',
  'frostbite': 'trauma_injury',
  'heat stroke': 'trauma_injury',
  'heat exhaustion': 'trauma_injury',
  'hypothermia': 'trauma_injury',
  'burn': 'trauma_injury',
  'systemic lupus erythematosis (sle)': 'autoimmune',
  'scleroderma': 'autoimmune',
  'sjogren syndrome': 'autoimmune',
  'vasculitis': 'autoimmune',
  'primary immunodeficiency': 'autoimmune',
  'amyloidosis': 'autoimmune',
  'carbon monoxide poisoning': 'poisoning',
  'drug reaction': 'poisoning',
  'drug poisoning due to medication': 'poisoning',
  'insulin overdose': 'poisoning',
  'vitamin a deficiency': 'nutritional',
  'vitamin b12 deficiency': 'nutritional',
  'vitamin d deficiency': 'nutritional',
  'scurvy': 'nutritional',
  'eating disorder': 'nutritional',
  'magnesium deficiency': 'nutritional',
  'folate deficiency': 'nutritional',
  'wilson disease': 'nutritional',
  'dental caries': 'dental_oral',
  'tooth abscess': 'dental_oral',
  'tooth disorder': 'dental_oral',
  'gum disease': 'dental_oral',
  'aphthous ulcer': 'dental_oral',
  'oral thrush (yeast infection)': 'dental_oral',
  'temporomandibular joint disorder': 'dental_oral',
  'jaw disorder': 'dental_oral',
};

export function getDiseaseGroup(diseaseName: string): string {
  const key = diseaseName.toLowerCase().trim();
  const override = DISEASE_GROUP_OVERRIDES_KB[key];
  if (override) return override;
  for (const [group, keywords] of GROUP_KEYWORD_RULES_TS) {
    if (keywords.some(kw => key.includes(kw))) return group;
  }
  return 'other';
}

export function getDiseasesByGroup(groupName: string): string[] {
  const diseases = medicalKnowledge as Disease[];
  return diseases
    .filter(d => getDiseaseGroup(d.disease) === groupName)
    .map(d => d.disease);
}

export function searchDiseasesInGroup(symptoms: string[], groupName: string): SearchResult[] {
  const groupNames = new Set(getDiseasesByGroup(groupName).map(d => d.toLowerCase()));
  const diseases = (medicalKnowledge as Disease[]).filter(
    d => groupNames.has(d.disease.toLowerCase())
  );

  if (diseases.length < 3) {
    return searchDiseases(symptoms);
  }

  const results: SearchResult[] = diseases.map(disease => {
    const relevance = calculateSimilarity(symptoms, disease.symptoms);
    const normalizedInput = symptoms.map(normalizeText);
    const matched = disease.symptoms.filter(ds =>
      normalizedInput.some(is => is.includes(normalizeText(ds)) || normalizeText(ds).includes(is))
    );
    const dsProfile = datasetService.getDiseaseProfile(disease.disease);
    return {
      disease: disease.disease,
      relevance,
      matchedSymptoms: matched,
      description: disease.description,
      treatments: disease.treatments,
      severity: disease.severity,
      datasetRecords: dsProfile?.totalRecords,
    };
  });

  return results
    .filter(r => r.relevance >= 0)
    .sort((a, b) => b.relevance - a.relevance)
    .slice(0, 10);
}

export function augmentPrompt(symptoms: string[], userQuery?: string): string {
  const topResults = searchDiseases(symptoms);
  
  let context = "Relevant medical information:\n\n";
  
  for (const result of topResults.slice(0, 5)) {
    context += `Disease: ${result.disease}\n`;
    context += `Description: ${result.description}\n`;
    context += `Symptoms: ${result.matchedSymptoms.join(', ')}\n`;
    context += `Treatments: ${result.treatments.join(', ')}\n`;
    context += `Severity: ${result.severity}\n`;
    if (result.datasetRecords) {
      context += `Dataset Records: ${result.datasetRecords} cases\n`;
    }
    context += "\n";
  }
  
  // Add structured dataset analysis context from CSV files
  const datasetContext = datasetService.buildDatasetRAGContext(symptoms);
  if (datasetContext) {
    context += "=== Dataset Analysis (RAG) ===\n";
    context += datasetContext;
    context += "\n";
  }
  
  if (userQuery) {
    context += `\nUser question: ${userQuery}\n`;
  }
  
  return context;
}
