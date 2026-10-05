import { useState, useRef, useEffect } from 'react';
import { Send, HelpCircle, CheckCircle, AlertCircle, Stethoscope, AlertTriangle, Sparkles } from 'lucide-react';
import { extractAndPredict, getFollowUpQuestions, submitFollowUpAnswers, saveToHistory } from '@/api/config';

export interface ChatMessage {
  id: number;
  role: 'ai' | 'user';
  type: 'text' | 'followup' | 'result_ready' | 'error';
  content: string;
  timestamp: Date;
  questionIndex?: number;
  answered?: string;
  symptoms?: string[];
}

interface Props {
  onViewResults: () => void;
  setPredictionResult: (result: Record<string, unknown>) => void;
}

const quickChips = [
  '🤒 Fever', '🤕 Headache', '😮‍💨 Shortness of breath', '🤢 Nausea',
  '😴 Fatigue', '💔 Chest Pain', '🦴 Joint Pain', '😵 Dizziness', '🤧 Runny Nose', '🥵 Chills'
];

export default function ChatScreen({ onViewResults, setPredictionResult }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [chatPhase, setChatPhase] = useState<'input' | 'followup' | 'done'>('input');
  const [isTyping, setIsTyping] = useState(false);
  const [typingText, setTypingText] = useState('');
  const [currentSymptoms, setCurrentSymptoms] = useState<string[]>([]);
  const [followUpQuestions, setFollowUpQuestions] = useState<string[]>([]);
  const [currentQIndex, setCurrentQIndex] = useState(0);
  const [followUpAnswers, setFollowUpAnswers] = useState<{ question: string; answer: string }[]>([]);
  const [followUpInput, setFollowUpInput] = useState('');
  const [isSubmittingAnswer, setIsSubmittingAnswer] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isTyping]);

  const addMsg = (msg: Omit<ChatMessage, 'id' | 'timestamp'>) => {
    setMessages(prev => [...prev, { ...msg, id: Date.now() + Math.random(), timestamp: new Date() }]);
  };

  const handleSend = async () => {
    if (!input.trim() || chatPhase !== 'input') return;
    const text = input.trim();
    setInput('');
    addMsg({ role: 'user', type: 'text', content: text });

    setIsTyping(true); setTypingText('Reading your symptoms...');
    const res = await extractAndPredict(text);
    setIsTyping(false);

    if (res.error) { addMsg({ role: 'ai', type: 'error', content: res.error }); return; }
    if (!res.symptoms_found?.length) {
      addMsg({ role: 'ai', type: 'text', content: "I couldn't identify specific symptoms from your description. Try being more specific, for example: 'I have had a fever of 38°C, headache, and body aches for 2 days.'" });
      return;
    }
    if (res.symptoms_found.length < 3) {
      addMsg({ role: 'ai', type: 'text', content: `I found these symptoms: ${res.symptoms_found.join(', ')}. Please add at least 3 symptoms for a more reliable result.`, symptoms: res.symptoms_found });
      return;
    }

    setCurrentSymptoms(res.symptoms_found);
    addMsg({ role: 'ai', type: 'text', content: `Thanks for sharing. I found these symptoms: ${res.symptoms_found.join(', ')}. Let me ask a few follow-up questions to better understand.`, symptoms: res.symptoms_found });
    setChatPhase('followup');

    setIsTyping(true); setTypingText('Preparing follow-up questions...');
    const fRes = await getFollowUpQuestions(res.symptoms_found);
    setIsTyping(false);

    if (fRes.error) {
      addMsg({ role: 'ai', type: 'error', content: fRes.error });
      setChatPhase('input');
      return;
    }
    if (!Array.isArray(fRes.questions) || fRes.questions.length === 0) {
      addMsg({ role: 'ai', type: 'error', content: 'I could not prepare follow-up questions. Please start a new check and try again.' });
      setChatPhase('input');
      return;
    }
    setFollowUpQuestions(fRes.questions);
    setCurrentQIndex(0);
    addMsg({ role: 'ai', type: 'followup', content: fRes.questions[0], questionIndex: 0 });
  };

  const handleFollowUpAnswer = async (answer: string) => {
    if (isSubmittingAnswer) return;
    setIsSubmittingAnswer(true);
    const question = followUpQuestions[currentQIndex];
    const newAnswers = [...followUpAnswers, { question, answer }];
    setFollowUpAnswers(newAnswers);
    setFollowUpInput('');

    // Mark answered
    setMessages(prev => prev.map(m =>
      m.type === 'followup' && m.questionIndex === currentQIndex ? { ...m, answered: answer } : m
    ));

    const nextIdx = currentQIndex + 1;
    if (nextIdx < followUpQuestions.length) {
      setCurrentQIndex(nextIdx);
      setTimeout(() => {
        addMsg({ role: 'ai', type: 'followup', content: followUpQuestions[nextIdx], questionIndex: nextIdx });
        setIsSubmittingAnswer(false);
      }, 500);
    } else {
      addMsg({ role: 'ai', type: 'text', content: "Got it! I have all the information I need. Let me analyze everything now..." });
      setChatPhase('done');
      setIsTyping(true); setTypingText('Analyzing with ML model...');
      const res = await submitFollowUpAnswers(currentSymptoms, newAnswers);
      setIsTyping(false);
      if (res.error) { addMsg({ role: 'ai', type: 'error', content: res.error }); setIsSubmittingAnswer(false); return; }
      
      setPredictionResult({ ...res, follow_up_answers: newAnswers });
      
      // Save to history
      const saveRes = await saveToHistory({
        disease: res.disease,
        confidence: res.confidence,
        symptoms_found: currentSymptoms,
        explanation: res.explanation,
        allPredictions: res.all_predictions,
      });
      if (saveRes.error) {
        console.warn('Save history failed:', saveRes.error);
      } else {
        addMsg({ role: 'ai', type: 'text', content: `Analysis saved to your history! (pattern score: ${res.confidence}%)` });
      }
      
      addMsg({ role: 'ai', type: 'result_ready', content: 'Your results are ready. Click below to view your full report.' });
    }
  };

  const autoResize = () => {
    const ta = textareaRef.current;
    if (ta) { ta.style.height = 'auto'; ta.style.height = Math.min(ta.scrollHeight, 120) + 'px'; }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Chat header */}
      <div className="bg-card border-b border-border px-5 py-3 flex items-center gap-3 shrink-0">
        <div className="w-9 h-9 rounded-full bg-primary-light flex items-center justify-center">
          <Stethoscope className="w-5 h-5 text-primary" />
        </div>
        <div>
          <div className="font-heading font-bold text-sm flex items-center gap-2">
            MediCare AI <span className="w-2 h-2 rounded-full bg-secondary inline-block" />
          </div>
          <span className="text-xs text-muted-foreground">
            {chatPhase === 'input' ? 'Ready to help' : chatPhase === 'followup' ? 'Gathering details...' : 'Analysis complete'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="mx-auto max-w-2xl pt-8 sm:pt-12">
            <div className="mb-7 text-center">
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-[22px] bg-primary-light text-primary shadow-soft">
                <Stethoscope className="h-8 w-8" />
              </div>
              <h2 className="font-heading text-2xl font-extrabold">Let’s start with how you feel</h2>
              <p className="mx-auto mt-2 max-w-lg text-sm leading-6 text-muted-foreground">
                Include what you’re feeling, when it began, and anything that makes it better or worse. Pick a few suggestions or write in your own words.
              </p>
            </div>
            <div className="mb-3 flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-[0.16em] text-muted-foreground">
              <Sparkles className="h-4 w-4 text-primary" /> Common symptoms
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {quickChips.map(c => (
                <button key={c} onClick={() => setInput(prev => prev ? prev + ', ' + c.slice(2).trim() : c.slice(2).trim())}
                  className="rounded-full border border-border bg-card px-3.5 py-2 text-xs font-semibold shadow-soft transition-all hover:-translate-y-0.5 hover:border-primary/40 hover:bg-primary-light hover:text-primary">
                  {c}
                </button>
              ))}
            </div>
            <p className="mt-3 text-center text-xs text-muted-foreground">Tap to add a symptom to your message — nothing is sent until you press the arrow.</p>
          </div>
        )}

        {messages.map(m => (
          <div key={m.id} className={`chat-bubble-enter flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {m.role === 'ai' && (
              <div className="w-8 h-8 rounded-full bg-primary-light flex items-center justify-center shrink-0 mr-2 mt-1">
                <Stethoscope className="w-4 h-4 text-primary" />
              </div>
            )}
            <div className={`max-w-[75%] ${m.role === 'user'
              ? 'gradient-user-bubble text-white rounded-[18px_18px_4px_18px] px-4 py-3'
              : m.type === 'followup'
                ? 'bg-card border-2 border-primary-light rounded-2xl p-5'
                : m.type === 'result_ready'
                  ? 'bg-card border-2 border-secondary rounded-2xl p-5'
                  : m.type === 'error'
                    ? 'bg-destructive/10 rounded-2xl p-4'
                    : 'bg-card border border-border rounded-[18px_18px_18px_4px] px-4 py-3'
              }`}>
              {m.type === 'followup' ? (
                <>
                  <div className="flex items-center gap-1.5 text-primary font-heading font-bold text-sm mb-2">
                    <HelpCircle className="w-4 h-4" /> Follow-up Question
                  </div>
                  <p className="text-[15px] mb-3">{m.content}</p>
                  {m.answered ? (
                    <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${m.answered === 'yes' ? 'bg-secondary/30 text-secondary-foreground' : 'bg-muted text-muted-foreground'}`}>
                      Your answer: {m.answered}
                    </span>
                  ) : (
                    <div className="space-y-3">
                      <div className="flex flex-wrap gap-2">
                        <button onClick={() => handleFollowUpAnswer('Yes')} disabled={isSubmittingAnswer} className="btn-pill bg-secondary/30 text-secondary-foreground px-6 py-2 text-sm hover:bg-secondary/50 disabled:opacity-50">Yes</button>
                        <button onClick={() => handleFollowUpAnswer('No')} disabled={isSubmittingAnswer} className="btn-pill bg-surface2 text-muted-foreground px-6 py-2 text-sm hover:bg-muted disabled:opacity-50">No</button>
                        <button onClick={() => handleFollowUpAnswer('Not sure')} disabled={isSubmittingAnswer} className="btn-pill border border-border bg-card px-4 py-2 text-sm text-muted-foreground hover:bg-muted disabled:opacity-50">Not sure</button>
                      </div>
                      <form onSubmit={event => { event.preventDefault(); if (followUpInput.trim()) handleFollowUpAnswer(followUpInput.trim()); }} className="flex gap-2">
                        <input
                          value={followUpInput}
                          onChange={event => setFollowUpInput(event.target.value)}
                          placeholder="Or type a more detailed answer"
                          aria-label="Your follow-up answer"
                          disabled={isSubmittingAnswer}
                          className="input-medicare min-w-0 flex-1 pl-4 py-2.5 text-sm"
                        />
                        <button type="submit" disabled={!followUpInput.trim() || isSubmittingAnswer} className="btn-primary px-4 py-2 text-sm disabled:opacity-50">
                          Answer
                        </button>
                      </form>
                    </div>
                  )}
                </>
              ) : m.type === 'result_ready' ? (
                <>
                  <div className="flex items-center gap-1.5 text-secondary font-heading font-bold text-base mb-2">
                    <CheckCircle className="w-5 h-5" /> Analysis Complete!
                  </div>
                  <p className="text-sm text-muted-foreground mb-3">{m.content}</p>
                  <button onClick={onViewResults} className="btn-primary w-full py-2.5 text-sm">View Full Report →</button>
                </>
              ) : m.type === 'error' ? (
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-destructive shrink-0 mt-0.5" />
                  <span className="text-destructive text-sm">{m.content}</span>
                </div>
              ) : (
                <>
                  <p className="text-[15px] leading-relaxed">{m.content}</p>
                  {m.symptoms && (
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {m.symptoms.map(s => (
                        <span key={s} className="bg-primary-light text-primary text-xs px-2.5 py-1 rounded-full font-semibold">{s}</span>
                      ))}
                    </div>
                  )}
                </>
              )}
              <div className="text-[11px] text-muted-foreground mt-1.5">
                {m.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex items-start gap-2 chat-bubble-enter">
            <div className="w-8 h-8 rounded-full bg-primary-light flex items-center justify-center shrink-0">
              <Stethoscope className="w-4 h-4 text-primary" />
            </div>
            <div className="bg-card border border-border rounded-[18px_18px_18px_4px] px-4 py-3">
              <div className="flex gap-1 mb-1">
                <span className="typing-dot" /><span className="typing-dot" /><span className="typing-dot" />
              </div>
              <span className="text-xs text-muted-foreground">{typingText}</span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      {chatPhase === 'input' && (
        <div className="border-t border-border bg-card px-5 py-4 shrink-0">
          <div className="flex items-end gap-3">
            <textarea ref={textareaRef} value={input} onChange={e => { setInput(e.target.value); autoResize(); }}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
              placeholder="What symptoms are you experiencing, and when did they start?"
              rows={1} className="input-medicare pl-4 resize-none flex-1" />
            <button onClick={handleSend} disabled={!input.trim()}
              className="btn-primary p-3 disabled:opacity-50">
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="mt-2 flex items-center justify-center gap-1.5 text-center text-[11px] leading-5 text-muted-foreground">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-accent-foreground" />
            If symptoms are severe or sudden, or include chest pain or trouble breathing, contact local emergency services.
          </p>
          <p className="mt-1 text-center text-[11px] text-muted-foreground">This tool is informational and is not a medical diagnosis.</p>
        </div>
      )}
    </div>
  );
}
