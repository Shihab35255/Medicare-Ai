import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Check,
  ClipboardCheck,
  HeartPulse,
  MessageCircle,
  ShieldCheck,
  Sparkles,
  Stethoscope,
} from 'lucide-react';

interface Props {
  onGetStarted: () => void;
  onSignIn: () => void;
  onAbout: () => void;
  onDemo?: () => void;
}

const steps = [
  {
    num: '01',
    icon: MessageCircle,
    title: 'Tell us how you feel',
    desc: 'Describe what you are experiencing in your own words. No medical terminology needed.',
  },
  {
    num: '02',
    icon: ClipboardCheck,
    title: 'Answer a few questions',
    desc: 'Helpful follow-up questions add context to the symptoms you share.',
  },
  {
    num: '03',
    icon: Sparkles,
    title: 'Explore your results',
    desc: 'Review possible matches and a clear explanation you can discuss with a clinician.',
  },
];

export default function LandingScreen({ onGetStarted, onSignIn, onAbout, onDemo }: Props) {
  return (
    <div className="landing-page screen-fade min-h-screen">
      <nav className="sticky top-0 z-50 border-b border-border/70 bg-card/85 backdrop-blur-xl">
        <div className="mx-auto flex h-[72px] max-w-7xl items-center justify-between px-5 sm:px-8">
          <a href="#" onClick={event => event.preventDefault()} className="flex items-center gap-2.5">
            <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-soft">
              <Stethoscope className="h-5 w-5" />
            </span>
            <span className="font-heading text-lg font-extrabold tracking-tight text-foreground">
              MediCare <span className="text-primary-dark">AI</span>
            </span>
          </a>
          <div className="flex items-center gap-2 sm:gap-3">
            <button onClick={onAbout} className="hidden rounded-full px-4 py-2 text-sm font-semibold text-muted-foreground transition hover:bg-muted hover:text-foreground sm:inline-flex">
              About
            </button>
            <button onClick={onSignIn} className="rounded-full px-3 py-2 text-sm font-semibold text-foreground transition hover:bg-muted sm:px-4">
              Sign in
            </button>
            <button onClick={onGetStarted} className="btn-primary flex items-center gap-2 px-4 py-2.5 text-sm sm:px-5">
              Get started <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </nav>

      <main>
        <section className="landing-grid relative isolate overflow-hidden">
          <div className="landing-glow landing-glow-one" />
          <div className="landing-glow landing-glow-two" />
          <div className="mx-auto grid max-w-7xl items-center gap-12 px-5 py-16 sm:px-8 sm:py-20 lg:grid-cols-[1.05fr_0.95fr] lg:gap-16 lg:py-24">
            <div className="relative z-10 text-center lg:text-left">
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/15 bg-card/80 px-4 py-2 text-sm font-semibold text-primary-dark shadow-soft">
                <span className="h-2 w-2 rounded-full bg-secondary" />
                A clearer first step toward understanding your health
              </div>
              <h1 className="mx-auto max-w-2xl font-heading text-[2.7rem] font-extrabold leading-[1.08] tracking-tight text-foreground sm:text-6xl lg:mx-0">
                Make sense of your
                <span className="gradient-text block pb-1">symptoms, together.</span>
              </h1>
              <p className="mx-auto mt-6 max-w-xl text-base leading-7 text-muted-foreground sm:text-lg sm:leading-8 lg:mx-0">
                Share what you’re feeling, answer thoughtful follow-up questions, and explore an easy-to-understand health summary.
              </p>
              <div className="mt-8 flex flex-col justify-center gap-3 sm:flex-row lg:justify-start">
                <button onClick={onGetStarted} className="btn-primary flex items-center justify-center gap-2 px-7 py-3.5 text-base shadow-card">
                  Start a symptom check <ArrowRight className="h-5 w-5" />
                </button>
                <a href="#how-it-works" className="btn-ghost flex items-center justify-center gap-2 px-7 py-3.5 text-base">
                  How it works <ArrowUpRight className="h-4 w-4" />
                </a>
              </div>
              {onDemo && (
                <button onClick={onDemo} className="mt-4 rounded-full px-3 py-2 text-sm font-semibold text-primary-dark transition hover:bg-primary-light">
                  Try the interactive demo
                </button>
              )}
              <div className="mt-9 flex flex-wrap justify-center gap-x-5 gap-y-3 text-sm text-muted-foreground lg:justify-start">
                <span className="inline-flex items-center gap-2"><Check className="h-4 w-4 text-primary" /> Plain-language guidance</span>
                <span className="inline-flex items-center gap-2"><Check className="h-4 w-4 text-primary" /> Your check-in history</span>
              </div>
            </div>

            <div className="relative mx-auto w-full max-w-[510px] lg:mr-4">
              <div className="absolute -inset-5 rounded-[3rem] bg-gradient-to-br from-primary-light via-transparent to-purple/30 blur-2xl" />
              <div className="relative rounded-[2rem] border border-white/80 bg-white/75 p-3 shadow-card backdrop-blur-xl sm:p-5">
                <div className="rounded-[1.5rem] border border-border/70 bg-card p-5 sm:p-7">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary-light text-primary-dark">
                        <HeartPulse className="h-6 w-6" />
                      </div>
                      <div>
                        <p className="font-heading font-bold">Your health check-in</p>
                        <p className="text-xs text-muted-foreground">A helpful place to start</p>
                      </div>
                    </div>
                    <span className="rounded-full bg-secondary/25 px-3 py-1 text-xs font-bold text-secondary-foreground">Guided</span>
                  </div>

                  <div className="mt-7 rounded-2xl bg-muted/80 p-4 sm:p-5">
                    <p className="text-xs font-bold uppercase tracking-[0.16em] text-muted-foreground">Your conversation</p>
                    <div className="mt-4 flex justify-end">
                      <div className="max-w-[90%] rounded-[18px_18px_4px_18px] bg-primary px-4 py-3 text-sm leading-6 text-primary-foreground shadow-soft">
                        I’ve had a headache since yesterday. What details should I share?
                      </div>
                    </div>
                    <div className="mt-3 flex items-start gap-2.5">
                      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-card text-primary shadow-soft">
                        <Sparkles className="h-4 w-4" />
                      </span>
                      <div className="max-w-[90%] rounded-[18px_18px_18px_4px] border border-border/70 bg-card px-4 py-3 text-sm leading-6 text-foreground">
                        Tell me what else you’re experiencing, when it started, and how it feels. I’ll ask a few follow-up questions.
                      </div>
                    </div>
                  </div>

                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-2xl border border-border/70 bg-card p-3.5">
                      <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-secondary/25 text-secondary-foreground">
                        <Activity className="h-4 w-4" />
                      </div>
                      <p className="text-sm font-bold">Symptom insights</p>
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">A structured summary to review</p>
                    </div>
                    <div className="rounded-2xl border border-border/70 bg-card p-3.5">
                      <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-xl bg-purple/30 text-foreground">
                        <ShieldCheck className="h-4 w-4" />
                      </div>
                      <p className="text-sm font-bold">Your next step</p>
                      <p className="mt-1 text-xs leading-5 text-muted-foreground">Always guided by real care</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="hero-float absolute -left-5 top-[22%] hidden items-center gap-2 rounded-2xl border border-white/80 bg-card px-4 py-3 shadow-card sm:flex">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-secondary/25 text-secondary-foreground"><Check className="h-5 w-5" /></span>
                <span className="text-sm font-bold">Simple, guided questions</span>
              </div>
              <div className="hero-float-delayed absolute -right-3 bottom-[12%] hidden items-center gap-2 rounded-2xl border border-white/80 bg-card px-4 py-3 shadow-card sm:flex">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-light text-primary"><ClipboardCheck className="h-5 w-5" /></span>
                <span className="text-sm font-bold">Keep track over time</span>
              </div>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="mx-auto max-w-7xl px-5 py-16 sm:px-8 sm:py-20">
          <div className="mx-auto mb-10 max-w-2xl text-center">
            <span className="text-xs font-extrabold uppercase tracking-[0.2em] text-primary-dark">A simple process</span>
            <h2 className="mt-3 font-heading text-3xl font-extrabold tracking-tight sm:text-4xl">From symptoms to a clearer picture</h2>
            <p className="mt-3 text-muted-foreground">A few guided steps help you organize what you’re experiencing.</p>
          </div>
          <div className="grid gap-4 sm:grid-cols-3 sm:gap-5">
            {steps.map(({ num, icon: Icon, title, desc }) => (
              <article key={num} className="card-medicare group relative overflow-hidden p-6 transition duration-300 hover:-translate-y-1 hover:shadow-card sm:p-7">
                <span className="absolute right-5 top-4 font-heading text-4xl font-extrabold text-primary/10">{num}</span>
                <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary-light text-primary transition duration-300 group-hover:scale-105">
                  <Icon className="h-6 w-6" />
                </span>
                <h3 className="mt-5 font-heading text-lg font-extrabold">{title}</h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{desc}</p>
              </article>
            ))}
          </div>
          <div className="mt-8 flex flex-col items-center justify-between gap-4 rounded-3xl border border-primary/10 bg-gradient-to-r from-primary-light/70 via-card to-purple/20 p-6 sm:flex-row sm:px-8">
            <div>
              <p className="font-heading text-lg font-extrabold">Ready to start a check-in?</p>
              <p className="mt-1 text-sm text-muted-foreground">It only takes a few minutes to describe what’s going on.</p>
            </div>
            <button onClick={onGetStarted} className="btn-primary flex shrink-0 items-center gap-2 px-6 py-3">
              Get started <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </section>
      </main>

      <footer className="border-t border-border/70 bg-card/60 px-5 py-7 text-center sm:px-8">
        <p className="mx-auto flex max-w-3xl items-start justify-center gap-2 text-xs leading-5 text-muted-foreground">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
          MediCare AI is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment.
        </p>
        <p className="mt-3 text-xs text-muted-foreground">© {new Date().getFullYear()} MediCare AI</p>
      </footer>
    </div>
  );
}
