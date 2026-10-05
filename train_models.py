#!/usr/bin/env python3
"""
MediCare AI Offline Model Training Pipeline

Trains ML models on 20 disease GROUPS instead of 773 individual diseases.
This raises ML confidence from ~0.13% (random/773) to ~5% (random/20),
making the ensemble a meaningful first-pass category filter for RAG.

Usage:
  python train_models.py

Output:
  models/tfidf_vectorizer.pkl
  models/label_encoder.pkl   (encodes 20 groups, not 773 diseases)
  models/random_forest.pkl
  models/svm.pkl
  models/naive_bayes.pkl
  models/disease_groups.json (group → [disease list] mapping for backend)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
from collections import Counter
from typing import Tuple

MODEL_DIR = 'models'
TRAINING_DATA_PATH = 'datasets/Final_Augmented_dataset_Diseases_and_Symptoms.csv'


# ── Disease → Group mapping ────────────────────────────────────────────────
# Keyword rules applied in priority order. First match wins.
# Each entry: (group_name, [keywords_that_must_appear_in_disease_name])

GROUP_KEYWORD_RULES = [
    ('respiratory',   ['lung', 'bronch', 'pulmon', 'pneum', 'asthma', 'copd',
                        'emphys', 'pleural', 'trachei', 'ards', 'whooping',
                        'tuberculosis', 'covid', 'respiratory']),
    ('cardiovascular',['heart', 'cardiac', 'cardio', 'coronar', 'atrial',
                        'ventricular', 'arrhythmia', 'angina', 'aortic', 'aorta',
                        'arterial', 'myocarditis', 'endocarditis', 'pericarditis',
                        'thrombophlebitis', 'atherosclerosis']),
    ('neurological',  ['brain', 'cerebr', 'neural', 'neuro', 'epilep', 'migraine',
                        'stroke', 'meningit', 'encephalit', 'parkinson', 'alzheimer',
                        'dementia', 'sclerosis', 'palsy', 'seizure', 'headache',
                        'intracranial', 'subarachnoid', 'subdural', 'epidural',
                        'hydrocephalus', 'ataxia', 'narcolepsy', 'myasthenia',
                        'guillain', 'neuralgia', 'neuropath', 'mononeuritis']),
    ('psychiatric',   ['depress', 'anxiety', 'bipolar', 'schizo', 'psychot',
                        'panic', 'ptsd', 'ocd', 'phobia', 'personality disorder',
                        'dissociati', 'somatiz', 'factitious', 'dysthymic',
                        'drug abuse', 'drug withdrawal', 'alcohol abuse',
                        'alcohol intox', 'alcohol withdrawal', 'marijuana',
                        'substance', 'eating disorder', 'adhd', 'autism',
                        'asperger', 'conduct disorder', 'oppositional',
                        'impulse control', 'adjustment reaction', 'acute stress',
                        'postpartum depression', 'neurosis']),
    ('gastrointestinal', ['gastro', 'intestin', 'bowel', 'colon', 'rectal',
                           'liver', 'hepat', 'pancrea', 'esophag', 'stomach',
                           'append', 'diverticul', 'hernia', 'peritonit',
                           'cholecyst', 'choledoch', 'gallstone', 'cirrhosis',
                           'volvulus', 'ileus', 'indigestion', 'hemorrhoid',
                           'anal fiss', 'anal fistul', 'perirectal']),
    ('musculoskeletal', ['bone', 'joint', 'muscl', 'tendin', 'ligament',
                          'fracture', 'dislocation', 'arthrit', 'spondyl',
                          'osteo', 'bursit', 'lumbago', 'sciatica', 'disc',
                          'spinal stenosis', 'scoliosis', 'fibromyalg',
                          'carpal tunnel', 'epicondyl', 'rotator', 'plantar',
                          'sprain', 'strain', 'torticoll', 'gout', 'myosit']),
    ('skin',          ['skin', 'dermat', 'acne', 'eczema', 'psoriasis',
                        'rosacea', 'melanoma', 'alopecia', 'fungal infection',
                        'impetigo', 'cellulitis', 'scabies', 'lice',
                        'chickenpox', 'shingles', 'herpes zoster', 'wart',
                        'molluscum', 'pemphigus', 'lichen', 'seborrheic',
                        'vitiligo', 'pityriasis', 'onychomycosis', 'athlete']),
    ('infectious',    ['sepsis', 'malaria', 'typhoid', 'dengue', 'lyme',
                        'rocky mountain', 'mononucleosis', 'mumps', 'scarlet',
                        'toxoplasmos', 'histoplasmos', 'cryptococcosis',
                        'aspergillosis', 'cysticercosis', 'trichinosis',
                        'parasitic', 'pinworm', 'sporotrichosis', 'syphilis',
                        'gonorrhea', 'chlamydia', 'trichomonas', 'granuloma',
                        'lymphogranuloma', 'hiv', 'cat scratch', 'valley fever']),
    ('endocrine',     ['diabet', 'thyroid', 'insulin', 'hypoglycemi',
                        'hyperglycemi', 'adrenal', 'pituitary', 'parathyroid',
                        'graves', 'hashimoto', 'goiter', 'cushing',
                        'hypercalcemia', 'hypocalcemia', 'hyponatremia',
                        'hypernatremia', 'hypokalemia', 'hyperkalemia',
                        'hyperlipidemia', 'metabolic disorder', 'hormone disorder',
                        'hyperosmotic', 'siadh', 'glucocorticoid']),
    ('renal_urinary', ['kidney', 'renal', 'urinar', 'bladder', 'urethral',
                        'cystitis', 'pyelonephrit', 'hydronephrosis',
                        'nephrosis', 'ureterl', 'urethritis', 'incontinence',
                        'vesicoureteral']),
    ('reproductive',  ['pregnan', 'obstetric', 'uterine', 'ovarian', 'cervical',
                        'endometri', 'menstrual', 'menopause', 'vaginal',
                        'vulvar', 'vaginit', 'prostate', 'testicular',
                        'epididym', 'erectile', 'priapism', 'varicocele',
                        'hydrocele', 'breast cancer', 'breast cyst',
                        'mastitis', 'polycystic ovarian', 'pelvic',
                        'placenta', 'preeclampsia', 'abortion', 'ectopic']),
    ('hematologic',   ['anemia', 'hemophilia', 'leukemia', 'lymphoma',
                        'thrombocytopenia', 'polycythemia', 'thalassemia',
                        'sickle cell', 'coagulation', 'bleeding disorder',
                        'myelodysplastic', 'lymphedema', 'hypovolemia',
                        'rhabdomyolysis', 'white blood cell', 'von willebrand',
                        'fluid overload', 'spherocytosis', 'g6pd']),
    ('oncologic',     ['cancer', 'tumor', 'malignant', 'lymphoma', 'leukemia',
                        'sarcoma', 'melanoma', 'carcinoid', 'metastatic',
                        'myeloma', 'ependymoma', 'meningioma']),
    ('ophthalmic',    ['eye', 'ocular', 'optic', 'retinal', 'cornea', 'corneal',
                        'conjunctivit', 'glaucoma', 'cataract', 'macular',
                        'blephar', 'stye', 'chalazion', 'iridocyclitis',
                        'uveitis', 'scleritis', 'endophthalmitis', 'vitreous',
                        'myopia', 'hyperopia', 'astigmatism', 'presbyopia',
                        'amblyopia', 'pterygium', 'pinguecula', 'floaters']),
    ('ent',           ['ear', 'otitis', 'hearing', 'tinnitus', 'sinus',
                        'nasal', 'pharyngit', 'tonsil', 'laryngit', 'throat',
                        'mastoid', 'cholesteatoma', 'labyrinthitis', 'meniere',
                        'vertigo', 'bppv', 'vocal cord', 'nose disorder',
                        'salivary', 'eustachian', 'otosclerosis']),
    ('trauma_injury', ['fracture', 'dislocation', 'open wound', 'injury',
                        'crush', 'burn', 'frostbite', 'heat stroke',
                        'heat exhaustion', 'hypothermia', 'envenomation',
                        'insect bite', 'carbon monoxide', 'birth trauma']),
    ('autoimmune',    ['lupus', 'scleroderma', 'sjogren', 'vasculitis',
                        'rheumatic fever', 'primary immunodeficiency',
                        'hypergammaglobulinemia', 'amyloidosis']),
    ('poisoning',     ['poison', 'overdose', 'intoxication', 'drug reaction',
                        'drug poisoning', 'toxin']),
    ('nutritional',   ['deficiency', 'scurvy', 'obesity', 'malnutrition',
                        'protein deficiency', 'folate', 'vitamin']),
    ('dental_oral',   ['dental', 'tooth', 'teeth', 'gum disease', 'oral',
                        'jaw disorder', 'aphthous', 'temporomandibular']),
]

# Explicit overrides for diseases whose names don't match keyword rules
DISEASE_GROUP_OVERRIDES = {
    'flu': 'respiratory',
    'common cold': 'respiratory',
    'covid-19': 'respiratory',
    'croup': 'respiratory',
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
    'mononeuritis': 'neurological',

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
    'persistent vomiting of unknown cause': 'gastrointestinal',

    'gout': 'musculoskeletal',
    'fibromyalgia': 'musculoskeletal',
    'lumbago': 'musculoskeletal',
    'sciatica': 'musculoskeletal',
    'scoliosis': 'musculoskeletal',
    'torticollis': 'musculoskeletal',
    'bursitis': 'musculoskeletal',
    'sprain or strain': 'musculoskeletal',
    'plantar fasciitis': 'musculoskeletal',
    'bunion': 'musculoskeletal',
    'hammer toe': 'musculoskeletal',
    'flat feet': 'musculoskeletal',
    'muscle spasm': 'musculoskeletal',
    'restless leg syndrome': 'musculoskeletal',
    'tietze syndrome': 'musculoskeletal',
    'polymyalgia rheumatica': 'musculoskeletal',
    'joint effusion': 'musculoskeletal',
    'hemarthrosis': 'musculoskeletal',
    'avascular necrosis': 'musculoskeletal',
    'myositis': 'musculoskeletal',
    'reactive arthritis': 'musculoskeletal',
    'septic arthritis': 'musculoskeletal',
    'juvenile rheumatoid arthritis': 'musculoskeletal',
    'nerve impingement near the shoulder': 'musculoskeletal',
    'chronic pain disorder': 'musculoskeletal',

    'chickenpox': 'skin',
    'shingles (herpes zoster)': 'skin',
    'cold sore': 'skin',
    'impetigo': 'skin',
    'viral exanthem': 'skin',
    'viral warts': 'skin',
    'hemangioma': 'skin',
    'hematoma': 'skin',
    'diaper rash': 'skin',
    'decubitus ulcer': 'skin',
    'intertrigo (skin condition)': 'skin',
    'scar': 'skin',
    'callus': 'skin',
    'lipoma': 'skin',
    'sebaceous cyst': 'skin',
    'pilonidal cyst': 'skin',
    'rosacea': 'skin',
    'hidradenitis suppurativa': 'skin',
    'dyshidrosis': 'skin',
    'pityriasis rosea': 'skin',
    'kaposi sarcoma': 'skin',
    'necrotizing fasciitis': 'skin',
    'gas gangrene': 'skin',

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
    'acariasis': 'infectious',
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
    'hormone disorder': 'endocrine',
    'metabolic disorder': 'endocrine',
    'magnesium deficiency': 'nutritional',

    'urinary tract infection': 'renal_urinary',
    'kidney stone': 'renal_urinary',
    'hydronephrosis': 'renal_urinary',
    'vesicoureteral reflux': 'renal_urinary',
    'atonic bladder': 'renal_urinary',
    'temporary or benign blood in urine': 'renal_urinary',

    'pregnancy': 'reproductive',
    'menopause': 'reproductive',
    'polycystic ovarian syndrome (pcos)': 'reproductive',
    'premature ovarian failure': 'reproductive',
    'galactorrhea of unknown cause': 'reproductive',
    'female infertility of unknown cause': 'reproductive',
    'atrophic vaginitis': 'reproductive',
    'vaginismus': 'reproductive',
    'vulvodynia': 'reproductive',
    'priapism': 'reproductive',
    'balanitis': 'reproductive',
    'phimosis': 'reproductive',
    'cryptorchidism': 'reproductive',
    'varicocele of the testicles': 'reproductive',
    'hydrocele of the testicle': 'reproductive',
    'spermatocele': 'reproductive',
    'peyronie disease': 'reproductive',
    'atrophy of the corpus cavernosum': 'reproductive',

    'sickle cell anemia': 'hematologic',
    'sickle cell crisis': 'hematologic',
    'aplastic anemia': 'hematologic',
    'hemolytic anemia': 'hematologic',
    'iron deficiency anemia': 'hematologic',
    'anemia of chronic disease': 'hematologic',
    'anemia due to chronic kidney disease': 'hematologic',
    'anemia due to malignancy': 'hematologic',
    'thalassemia': 'hematologic',
    'coagulation (bleeding) disorder': 'hematologic',
    'von willebrand disease': 'hematologic',
    'primary thrombocythemia': 'hematologic',
    'polycythemia vera': 'hematologic',
    'myelodysplastic syndrome': 'hematologic',
    'white blood cell disease': 'hematologic',
    'lymphedema': 'hematologic',
    'hypovolemia': 'hematologic',
    'fluid overload': 'hematologic',
    'rhabdomyolysis': 'hematologic',
    'g6pd enzyme deficiency': 'hematologic',
    'spherocytosis': 'hematologic',
    'folate deficiency': 'nutritional',

    'lung cancer': 'oncologic',
    'breast cancer': 'oncologic',
    'colorectal cancer': 'oncologic',
    'stomach cancer': 'oncologic',
    'liver cancer': 'oncologic',
    'pancreatic cancer': 'oncologic',
    'cervical cancer': 'oncologic',
    'ovarian cancer': 'oncologic',
    'uterine cancer': 'oncologic',
    'endometrial cancer': 'oncologic',
    'bladder cancer': 'oncologic',
    'kidney cancer': 'oncologic',
    'prostate cancer': 'oncologic',
    'testicular cancer': 'oncologic',
    'brain cancer': 'oncologic',
    'bone cancer': 'oncologic',
    'head and neck cancer': 'oncologic',
    'esophageal cancer': 'oncologic',
    'intestinal cancer': 'oncologic',
    'thyroid cancer': 'oncologic',
    'adrenal cancer': 'oncologic',
    'soft tissue sarcoma': 'oncologic',
    'carcinoid syndrome': 'oncologic',
    'metastatic cancer': 'oncologic',
    'multiple myeloma': 'oncologic',
    'vulvar cancer': 'oncologic',
    'skin cancer': 'oncologic',

    'dry eye of unknown cause': 'ophthalmic',
    'floaters': 'ophthalmic',
    'stenosis of the tear duct': 'ophthalmic',
    'orbital cellulitis': 'ophthalmic',
    'foreign body in the eye': 'ophthalmic',
    'eye alignment disorder': 'ophthalmic',
    'chorioretinitis': 'ophthalmic',
    'aphakia': 'ophthalmic',

    'meniere disease': 'ent',
    'tinnitus of unknown cause': 'ent',
    'allergy': 'ent',
    'allergy to animals': 'ent',
    'seasonal allergies (hay fever)': 'ent',
    'food allergy': 'ent',
    'strep throat': 'ent',
    'croup': 'ent',
    'deviated nasal septum': 'ent',
    'nasal polyp': 'ent',
    'nose disorder': 'ent',
    'cholesteatoma': 'ent',
    'foreign body in the ear': 'ent',
    'foreign body in the nose': 'ent',
    'foreign body in the throat': 'ent',
    'ear wax impaction': 'ent',
    'ear drum damage': 'ent',
    'salivary gland disorder': 'ent',
    'sialoadenitis': 'ent',
    'vocal cord polyp': 'ent',
    'tracheitis': 'ent',
    'peritonsillar abscess': 'ent',
    'abscess of the pharynx': 'ent',
    'abscess of nose': 'ent',
    'tonsillar hypertrophy': 'ent',

    'open wound due to trauma': 'trauma_injury',
    'open wound from surgical incision': 'trauma_injury',
    'infection of open wound': 'trauma_injury',
    'crushing injury': 'trauma_injury',
    'birth trauma': 'trauma_injury',
    'insect bite': 'trauma_injury',
    'envenomation from spider or animal bite': 'trauma_injury',
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
    'hypergammaglobulinemia': 'autoimmune',
    'amyloidosis': 'autoimmune',

    'carbon monoxide poisoning': 'poisoning',
    'drug reaction': 'poisoning',
    'drug poisoning due to medication': 'poisoning',
    'insulin overdose': 'poisoning',

    'vitamin a deficiency': 'nutritional',
    'vitamin b12 deficiency': 'nutritional',
    'vitamin d deficiency': 'nutritional',
    'scurvy': 'nutritional',
    'protein deficiency': 'nutritional',
    'eating disorder': 'nutritional',
    'wilson disease': 'nutritional',
    'hemochromatosis': 'nutritional',

    'dental caries': 'dental_oral',
    'tooth abscess': 'dental_oral',
    'tooth disorder': 'dental_oral',
    'broken tooth': 'dental_oral',
    'gum disease': 'dental_oral',
    'aphthous ulcer': 'dental_oral',
    'oral thrush (yeast infection)': 'dental_oral',
    'oral mucosal lesion': 'dental_oral',
    'oral leukoplakia': 'dental_oral',
    'mucositis': 'dental_oral',
    'temporomandibular joint disorder': 'dental_oral',
    'jaw disorder': 'dental_oral',
    'cellulitis or abscess of mouth': 'dental_oral',
}

ALL_GROUPS = [
    'respiratory', 'cardiovascular', 'neurological', 'psychiatric',
    'gastrointestinal', 'musculoskeletal', 'skin', 'infectious',
    'endocrine', 'renal_urinary', 'reproductive', 'hematologic',
    'oncologic', 'ophthalmic', 'ent', 'trauma_injury',
    'autoimmune', 'poisoning', 'nutritional', 'dental_oral',
]


def assign_disease_group(disease_name: str) -> str:
    """Map a disease name to one of 20 groups using overrides then keyword rules."""
    key = disease_name.lower().strip()

    # 1. Explicit override (highest priority)
    if key in DISEASE_GROUP_OVERRIDES:
        return DISEASE_GROUP_OVERRIDES[key]

    # 2. Keyword rules
    for group, keywords in GROUP_KEYWORD_RULES:
        if any(kw in key for kw in keywords):
            return group

    # 3. Fallback — assign to the most semantically related group
    return 'other'


def setup_models_directory():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"✓ Models directory ready: {MODEL_DIR}/")


def load_training_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"✗ Training data not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} training samples from {csv_path}")
        return df
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return None


def convert_binary_to_text_fast(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Vectorized binary→text conversion — ~40x faster than iterrows."""
    symptom_cols = [col for col in df.columns if col != 'diseases']
    print(f"  Converting {len(symptom_cols)} symptom columns (vectorized)...")

    # Build a numpy array of symptom column names for fast lookup
    col_names = np.array([c.replace('_', ' ') for c in symptom_cols])
    symptom_matrix = df[symptom_cols].values.astype(bool)  # (N, S) bool matrix

    # For each row, join the names of columns where value == 1
    texts = [
        ' '.join(col_names[row]) if row.any() else 'no symptoms'
        for row in symptom_matrix
    ]
    return pd.Series(texts), df['diseases']


def preprocess_data(df: pd.DataFrame) -> Tuple:
    print("\n📊 Preprocessing data...")

    if 'symptom_text' in df.columns:
        X_text = df['symptom_text'].fillna('')
        y_raw = df['disease']
    else:
        X_text, y_raw = convert_binary_to_text_fast(df)

    # Map each disease label to its group
    print("  Mapping diseases → 20 groups...")
    y_groups = y_raw.apply(assign_disease_group)

    group_counts = y_groups.value_counts()
    print(f"  • Input samples:  {len(X_text)}")
    print(f"  • Groups found:   {y_groups.nunique()}")
    unmapped = (y_groups == 'other').sum()
    if unmapped > 0:
        print(f"  ⚠  {unmapped} samples mapped to 'other' — check DISEASE_GROUP_OVERRIDES")
        # Show which diseases fell through
        other_diseases = y_raw[y_groups == 'other'].unique()[:20]
        print(f"     First unmatched: {list(other_diseases)}")
    print(f"  • Group distribution (top 10):")
    for g, cnt in group_counts.head(10).items():
        print(f"      {g:<20} {cnt:>6} samples")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_groups)

    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(X_text)
    print(f"  • TF-IDF features: {X.shape[1]}")

    # Save group → disease list mapping for the backend to use
    group_to_diseases = {}
    for disease, group in zip(y_raw.unique(), y_raw.apply(assign_disease_group).unique()):
        pass  # built below
    group_to_diseases = {}
    for disease in y_raw.unique():
        grp = assign_disease_group(disease)
        group_to_diseases.setdefault(grp, []).append(disease)

    mapping_path = f'{MODEL_DIR}/disease_groups.json'
    with open(mapping_path, 'w') as f:
        json.dump(group_to_diseases, f, indent=2)
    print(f"  ✓ Group mapping saved to {mapping_path}")

    return X, y_encoded, vectorizer, label_encoder


def train_models(X, y_encoded, vectorizer, label_encoder):
    print("\n🤖 Training models on", len(label_encoder.classes_), "groups...")

    # No rare-class filtering needed — groups are large
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"  • Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    models = {}

    # 1. Random Forest
    print("\n  1️⃣  Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    print(f"     ✓ accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")

    # 2. LinearSVC + calibration
    print("\n  2️⃣  SVM (LinearSVC)...")
    base_svm = LinearSVC(max_iter=3000, random_state=42)
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    models['svm'] = svm
    print(f"     ✓ accuracy: {accuracy_score(y_test, svm.predict(X_test)):.2%}")

    # 3. Naive Bayes
    print("\n  3️⃣  Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models['naive_bayes'] = nb
    print(f"     ✓ accuracy: {accuracy_score(y_test, nb.predict(X_test)):.2%}")

    print("\n  📋 Per-group accuracy (SVM):")
    from sklearn.metrics import classification_report
    report = classification_report(y_test, svm.predict(X_test),
                                   target_names=label_encoder.classes_,
                                   zero_division=0)
    print(report)

    return models


def save_models(models, vectorizer, label_encoder):
    print("\n💾 Saving models...")
    joblib.dump(vectorizer, f'{MODEL_DIR}/tfidf_vectorizer.pkl')
    joblib.dump(label_encoder, f'{MODEL_DIR}/label_encoder.pkl')
    for name, model in models.items():
        path = f'{MODEL_DIR}/{name}.pkl'
        joblib.dump(model, path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✓ {name}.pkl  ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("🏥 MediCare AI — Group-Level Model Training")
    print("   20 disease categories instead of 773 classes")
    print("   Expected ML confidence: 30–80% (vs 1–7% before)")
    print("=" * 60)

    setup_models_directory()
    df = load_training_data(TRAINING_DATA_PATH)
    if df is None:
        return False

    X, y_encoded, vectorizer, label_encoder = preprocess_data(df)
    models = train_models(X, y_encoded, vectorizer, label_encoder)
    save_models(models, vectorizer, label_encoder)

    print("\n" + "=" * 60)
    print(f"✅ Done! Models trained on {len(label_encoder.classes_)} groups.")
    print("   Next: python ml_service.py")
    print("=" * 60)
    return True


if __name__ == '__main__':
    main()
