import React, { Suspense, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Radio, Sparkles } from 'lucide-react';
import { Button } from '../ui/Button';
import EarthScene from './EarthScene';

/** Respects the OS "reduce motion" setting for the orbit + spin animation. */
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReduced(query.matches);
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);

  return reduced;
}

const scrollTo = (id: string) => {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
};

export const Hero: React.FC = () => {
  const navigate = useNavigate();
  const reducedMotion = usePrefersReducedMotion();

  return (
    <section className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-ground">
      {/* Globe */}
      <div className="absolute inset-0 z-0">
        <Suspense fallback={<div className="h-full w-full bg-ground" />}>
          <EarthScene paused={reducedMotion} />
        </Suspense>
      </div>

      {/* A light wash behind the copy only — the globe sits low in frame, so
          it does not need to be hidden to keep the headline readable. */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 z-[1] h-[62%]"
        style={{
          background:
            'linear-gradient(to bottom, rgba(5,7,12,0.82) 0%, rgba(5,7,12,0.55) 45%, rgba(5,7,12,0) 100%)',
        }}
        aria-hidden
      />

      {/* Copy */}
      <div className="relative z-10 mx-auto max-w-3xl px-6 pb-[38vh] text-center">
        <motion.div
          initial={{ opacity: 0, y: 22 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        >
          <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-line-strong bg-surface-2/70 px-3.5 py-1.5 text-xs backdrop-blur-md">
            <Radio size={12} className="animate-pulse text-accent" aria-hidden />
            <span className="font-medium text-ink">Smart India Hackathon 2026</span>
            <span className="text-ink-faint" aria-hidden>
              ·
            </span>
            <span className="text-ink-muted">PS 26167</span>
            <span className="text-ink-faint" aria-hidden>
              ·
            </span>
            <span className="text-teal">ISRO</span>
          </div>

          <h1 className="mb-5 text-5xl font-semibold leading-[1.05] tracking-tight text-ink sm:text-6xl md:text-7xl">
            Diachron <span className="text-accent text-glow">AI</span>
          </h1>

          <p className="mx-auto mb-9 max-w-xl text-base leading-relaxed text-ink-muted sm:text-lg">
            Semantic retrieval and multi-temporal change intelligence for satellite archives.
            Search by open-vocabulary natural language, pinpoint earliest observation onset dates,
            and inspect verified change polygons on the interactive map.
          </p>

          <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Button size="lg" onClick={() => navigate('/console')}>
              <Sparkles size={16} />
              Open the workspace
            </Button>
            <Button variant="outline" size="lg" onClick={() => scrollTo('how-it-works')}>
              How it works
              <ArrowRight size={16} />
            </Button>
          </div>
        </motion.div>
      </div>

    </section>
  );
};

export default Hero;
