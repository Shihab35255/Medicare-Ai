import { useState, useEffect } from 'react';
import { ArrowLeft, Activity, Sparkles, AlertTriangle, CheckCircle, Save, RotateCcw, BarChart3, Book, Brain, Cpu, Check } from 'lucide-react';
import { saveToHistory } from '@/api/config';

interface PredictionCandidate {
  disease: string;
  confidence: number;
  source?: string;
}

interface ResultReport {
  error?: string;
  confidence?: number;
  disease?: string;
  prediction_source?: string;
  symptoms_found?: string[];
  all_predictions?: PredictionCandidate[];
  methodology?: {
    methods_used?: string[];
    explanation?: string;
    ml_models_used?: {
      rf_prob?: number;
      svm_prob?: number;
      nb_prob?: number;
    };
  };
  rag_validation?: {
    rag_score?: number;
    score?: number;
    matched?: string[];
    matchedSymptoms?: string[];
    missing?: string[];
    missingSymptoms?: string[];
  };
  explanation?: string;
  explanation_pending?: boolean;
  follow_up_answers?: Array<{ question: string; answer: string }>;
}

interface Props {
  result: ResultReport | null;
  user: { id: string } | null;
  onBack: () => void;
  onNewCheck: () => void;
  onDashboard: () => void;
  onToast: (msg: string, type: 'success' | 'error' | 'info') => void;
}

function renderExplanation(text: string) {
  return text.split(/\n\s*\n/).filter(Boolean).map((section, index) => {
    const heading = section.match(/^\*\*(.+?)\*\*\s*([\s\S]*)$/);
    const body = (heading ? heading[2] : section).replace(/\*\*/g, '');
    return (
      <p key={index} className="mb-3 last:mb-0">
        {heading && <span className="mb-1 block font-heading font-bold text-foreground">{heading[1]}</span>}
        {body}
      </p>
    );
  });
}

export default function ResultsScreen({ result, user, onBack, onNewCheck, onDashboard, onToast }: Props) {
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [barWidth, setBarWidth] = useState(0);

  useEffect(() => {
    if (typeof result?.confidence === 'number') {
      const timeoutId = setTimeout(() => setBarWidth(result.confidence), 100);
      return () => clearTimeout(timeoutId);
    }
  }, [result?.confidence]);

  if (!result || result.error) {
    return (
      <div className="screen-fade max-w-[720px] mx-auto px-5 py-10">
        <button onClick={onBack} className="flex items-center gap-1 text-muted-foreground text-sm mb-6 hover:text-foreground transition-colors">
          <ArrowLeft className="w-4 h-4" /> Back to Chat
        </button>
        <div className="text-center py-20">
          <AlertTriangle className="w-16 h-16 text-destructive mx-auto mb-4" />
          <h2 className="text-xl font-heading font-bold text-foreground mb-2">No Results Available</h2>
          <p className="text-muted-foreground mb-6">
            {result?.error || "It looks like the analysis results are not available. Please try running the analysis again."}
          </p>
          <button onClick={onNewCheck} className="btn-primary">Start New Check</button>
        </div>
      </div>
    );
  }

  const confidenceValue = typeof result.confidence === 'number' ? result.confidence : 0;
  const isWeakSignal = result.prediction_source === 'fallback' || result.disease === 'Unknown Condition';

  const mlBreakdown = result.methodology?.ml_models_used ?? {};
  const mlModels = [
    { key: 'rf_prob', label: 'Random Forest', confidence: mlBreakdown.rf_prob },
    { key: 'svm_prob', label: 'SVM', confidence: mlBreakdown.svm_prob },
    { key: 'nb_prob', label: 'Naive Bayes', confidence: mlBreakdown.nb_prob },
  ].filter((model): model is typeof model & { confidence: number } => typeof model.confidence === 'number');

  const ragScore = result.rag_validation?.rag_score ?? result.rag_validation?.score ?? 0;
  const ragMatched = result.rag_validation?.matched ?? result.rag_validation?.matchedSymptoms ?? [];
  const ragMissing = result.rag_validation?.missing ?? result.rag_validation?.missingSymptoms ?? [];

  const handleSave = async () => {
    if (!user) { onToast('Please sign in to save', 'info'); return; }
    setSaving(true);
    const res = await saveToHistory(result);
    setSaving(false);
    if (res.error) { onToast(res.error, 'error'); return; }
    setSaved(true);
    onToast('Saved to your history!', 'success');
  };

  const otherPredictions = (result.all_predictions || []).filter(prediction => prediction.source !== 'ml_group_filter');
  const followUpAnswers = Array.isArray(result.follow_up_answers) ? result.follow_up_answers : [];

  return (
    <div className="screen-fade max-w-[720px] mx-auto px-5 py-10">
      <button onClick={onBack} className="flex items-center gap-1 text-muted-foreground text-sm mb-6 hover:text-foreground transition-colors">
        <ArrowLeft className="w-4 h-4" /> Back to Chat
      </button>

      {/* Header */}
      <div className="gradient-header-card rounded-3xl p-8 mb-6">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-10 h-10 rounded-full bg-card flex items-center justify-center"><Activity className="w-5 h-5 text-primary" /></div>
          <span className="text-primary font-heading font-semibold text-sm">Analysis Complete</span>
        </div>
        <h1 className="font-heading font-extrabold text-2xl mb-3">Your Symptom Report</h1>
        <div className="flex flex-wrap gap-1.5 mb-2">
          {result.symptoms_found?.map((s: string) => (
            <span key={s} className="bg-primary-light text-primary text-xs px-2.5 py-1 rounded-full font-semibold">{s}</span>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">{new Date().toLocaleString()}</p>
      </div>

      {followUpAnswers.length > 0 && (
        <section className="card-medicare mb-4 p-5" aria-labelledby="follow-up-details-heading">
          <h2 id="follow-up-details-heading" className="font-heading font-bold text-base">Details you shared</h2>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            These answers are shown for you to review or discuss with a clinician. The current pattern score is based on the symptoms listed above; these follow-up answers are not used to calculate it.
          </p>
          <dl className="mt-4 space-y-3">
            {followUpAnswers.map((item, index) => (
              <div key={`${item.question}-${index}`} className="border-t border-border pt-3 first:border-0 first:pt-0">
                <dt className="text-sm font-semibold">{item.question}</dt>
                <dd className="mt-1 text-sm text-muted-foreground">{item.answer}</dd>
              </div>
            ))}
          </dl>
        </section>
      )}

      {/* Pattern match */}
      {!isWeakSignal ? (
        <div className="card-medicare border-2 border-primary-light p-7 mb-4">
          <div className="flex items-start justify-between flex-wrap gap-2 mb-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-wide text-muted-foreground">Closest symptom-pattern match</p>
              <h2 className="mt-1 font-heading font-extrabold text-2xl">{result.disease}</h2>
            </div>
            <span className="bg-primary-light text-primary font-heading font-bold text-sm px-3 py-1 rounded-full">{confidenceValue}% pattern score</span>
          </div>
          <div className="mt-5">
            <div className="flex justify-between text-sm mb-1.5">
              <span className="text-muted-foreground">Symptom-pattern score</span>
              <span className="font-heading font-bold">{confidenceValue}%</span>
            </div>
            <div className="h-2.5 bg-border rounded-full overflow-hidden">
              <div className="h-full rounded-full bg-gradient-to-r from-primary to-purple confidence-bar-fill" style={{ width: `${Math.min(Math.max(barWidth, 0), 100)}%` }} />
            </div>
          </div>
          <p className="mt-3 text-xs leading-5 text-muted-foreground">
            This is a software pattern-matching score, not a probability, diagnosis, or measure of how likely you have this condition.
          </p>
        </div>
      ) : (
        <div className="card-medicare border-2 border-red-200 bg-red-50/30 p-7 mb-4">
          <h2 className="font-heading font-extrabold text-xl text-red-700 mb-2">No clear symptom-pattern match found</h2>
          <p className="text-sm leading-6 text-muted-foreground">The information shared does not point clearly to one condition in the system’s reference data. This does not mean nothing is wrong. A healthcare professional can assess your symptoms in context.</p>
        </div>
      )}

      {/* Other pattern matches */}
      {(otherPredictions.length > 0 || isWeakSignal) && (
        <div className="mb-4">
          <h3 className="font-heading font-bold text-base mb-1">{isWeakSignal ? 'Patterns the system noticed' : 'Other pattern matches'}</h3>
          <p className="mb-3 text-xs leading-5 text-muted-foreground">These are comparisons from the reference data, not ranked probabilities or confirmed conditions.</p>
          <div className={`space-y-2 ${isWeakSignal ? 'opacity-70' : ''}`}>
            {(isWeakSignal ? result.all_predictions || [] : otherPredictions).map((p, i) => (
              <div key={i} className="card-medicare bg-surface2 p-4 flex items-center justify-between">
                <span className="font-heading font-semibold text-sm">{p.disease}</span>
                <div className="flex items-center gap-3">
                  <div className="w-24 h-2 bg-border rounded-full overflow-hidden">
                    <div className="h-full rounded-full confidence-bar-fill" style={{ width: `${p.confidence}%`, background: i === 0 ? 'hsl(var(--accent))' : 'hsl(var(--accent2))' }} />
                  </div>
                  <span className="text-xs font-semibold text-muted-foreground w-20 text-right">{p.confidence}% score</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prediction Methodology */}
      {result.methodology && (
        <details className="card-medicare group mb-4">
          <summary className="flex cursor-pointer list-none items-center gap-2 p-5 font-heading font-bold text-base marker:hidden">
            <BarChart3 className="h-5 w-5 text-accent" />
            How the system compared patterns
            <span className="ml-auto text-xs font-normal text-muted-foreground group-open:hidden">Optional details</span>
          </summary>
          <div className="border-t border-border p-5 pt-4">
          {/* Method badges */}
          <div className="flex flex-wrap gap-2 mb-3">
            {result.methodology.methods_used?.includes('rag') && (
              <span className="bg-primary-light/30 text-primary text-xs px-2.5 py-1.5 rounded-full font-semibold flex items-center gap-1.5">
                <Book className="w-3 h-3" />
                RAG Knowledge
              </span>
            )}
            {result.methodology.methods_used?.includes('llm') && (
              <span className="bg-purple/30 text-purple-foreground text-xs px-2.5 py-1.5 rounded-full font-semibold flex items-center gap-1.5">
                <Brain className="w-3 h-3" />
                LLM Analysis
              </span>
            )}
            {result.methodology.methods_used?.includes('ml') && (
              <span className="bg-accent/30 text-accent-foreground text-xs px-2.5 py-1.5 rounded-full font-semibold flex items-center gap-1.5">
                <Cpu className="w-3 h-3" />
                ML Ensemble
              </span>
            )}
          </div>

          {/* Plain English explanation */}
          <p className="text-sm text-muted-foreground leading-relaxed mb-3">
            {result.methodology.explanation}
          </p>

          {/* ML model details if ML was used */}
          {mlModels.length > 0 && (
            <>
              <div className="border-t border-border pt-3 mt-3">
                <p className="text-xs font-semibold text-muted-foreground mb-2">Technical model details:</p>
                <div className="space-y-2">
                  {mlModels.map(model => (
                    <div key={model.key} className="flex items-center justify-between text-sm">
                      <span className="capitalize text-muted-foreground">{model.label}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-border rounded-full overflow-hidden">
                          <div className="h-full rounded-full bg-secondary transition-all" style={{ width: `${Math.min(model.confidence ?? 0, 100)}%` }} />
                        </div>
                        <span className="text-xs w-8 text-right font-mono">{(model.confidence ?? 0).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {result.rag_validation && (
            <div className="border-t border-border pt-3 mt-3">
              <p className="text-xs font-semibold text-muted-foreground mb-2">Reference profile comparison:</p>
              <p className="text-sm mb-3">Pattern overlap score: <span className="font-semibold">{ragScore.toFixed(1)}%</span> <span className="text-xs text-muted-foreground">(not a probability)</span></p>
              <div className="grid gap-3 md:grid-cols-2 text-sm">
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Matching symptoms</p>
                  {ragMatched.length > 0 ? (
                    <ul className="list-disc list-inside text-muted-foreground">
                      {ragMatched.map((symptom: string) => (
                        <li key={symptom}>{symptom}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-muted-foreground">No matching symptoms found.</p>
                  )}
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Other symptoms in reference profile</p>
                  {ragMissing.length > 0 ? (
                    <ul className="list-disc list-inside text-muted-foreground">
                      {ragMissing.map((symptom: string) => (
                        <li key={symptom}>{symptom}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-muted-foreground">No additional profile symptoms to show.</p>
                  )}
                </div>
              </div>
            </div>
          )}
          </div>
        </details>
      )}

      {/* Explanation */}
      {result.disease && (
        <div className="card-medicare p-7 mb-4">
          <div className="flex items-center gap-2 mb-1">
            <Sparkles className="w-5 h-5 text-primary" />
            <h3 className="font-heading font-bold text-base">AI Explanation</h3>
            {result.explanation_pending && <span className="text-xs text-muted-foreground animate-pulse">Generating...</span>}
          </div>
          <p className="text-xs text-muted-foreground mb-3">AI-generated summary — may be incomplete or incorrect; not a diagnosis.</p>
          <div className="text-[15px] leading-[1.8] text-muted-foreground">
            {result.explanation
              ? renderExplanation(result.explanation)
              : <span className="italic text-muted-foreground">Detailed explanation is being generated...</span>}
          </div>
        </div>
      )}

      {/* Next steps */}
      <div className="gradient-nextsteps border border-accent rounded-2xl p-7 mb-4">
        <h3 className="font-heading font-bold text-base mb-3">Helpful next steps</h3>
        <ul className="space-y-2 text-sm text-muted-foreground">
          <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" /> Note when symptoms began, how they have changed, and what makes them better or worse.</li>
          <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" /> Share your symptoms, existing conditions, allergies, and current medicines with a qualified healthcare professional.</li>
          <li className="flex gap-2"><Check className="mt-0.5 h-4 w-4 shrink-0 text-primary" /> Ask a clinician or pharmacist before starting, stopping, or changing any medicine.</li>
        </ul>
      </div>

      {/* Disclaimer */}
      <div className="bg-[#fff3cd] border border-[#ffc107] rounded-xl p-4 mb-6 flex items-start gap-2">
        <AlertTriangle className="w-5 h-5 text-[#856404] shrink-0 mt-0.5" />
        <div>
          <p className="font-heading font-bold text-sm text-[#856404] mb-1">When to get urgent help</p>
          <p className="text-xs text-[#856404] leading-relaxed">
            If you have severe trouble breathing, chest pain, fainting, sudden confusion, signs of a stroke, or another rapidly worsening or life-threatening symptom, contact your local emergency service now. Do not wait for an AI result.
          </p>
          <p className="mt-2 text-xs leading-relaxed text-[#856404]">This AI tool cannot diagnose or recommend treatment. For personal medical advice, contact a qualified healthcare professional.</p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-3">
        <button onClick={handleSave} disabled={saved || saving} className={`btn-pill px-5 py-2.5 text-sm flex items-center gap-2 ${saved ? 'bg-secondary/30 text-secondary-foreground' : 'btn-primary'}`}>
          {saved ? <><CheckCircle className="w-4 h-4" /> Saved</> : saving ? 'Saving...' : <><Save className="w-4 h-4" /> Save to History</>}
        </button>
        <button onClick={onNewCheck} className="btn-ghost px-5 py-2.5 text-sm flex items-center gap-2"><RotateCcw className="w-4 h-4" /> New Check</button>
        <button onClick={onDashboard} className="btn-ghost px-5 py-2.5 text-sm flex items-center gap-2"><BarChart3 className="w-4 h-4" /> View Dashboard</button>
      </div>
    </div>
  );
}
