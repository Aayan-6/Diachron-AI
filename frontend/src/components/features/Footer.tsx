import { Satellite, Radio } from 'lucide-react';

export const Footer = () => {
  return (
    <footer className="py-12 bg-space-black border-t border-white/10 text-xs">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6 pb-8 border-b border-white/5">
          {/* Branding & Attribution */}
          <div className="text-center md:text-left">
            <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
              <Satellite size={18} className="text-accent-cyan" />
              <h3 className="text-base font-bold text-white tracking-tight">
                Diachron <span className="text-accent-cyan">AI</span>
              </h3>
            </div>
            <p className="text-slate-400 max-w-md leading-relaxed">
              Autonomous diachronic analysis and multi-temporal change intelligence for Earth observation satellite archives.
            </p>
            <div className="flex flex-wrap items-center justify-center md:justify-start gap-2 mt-2 text-[11px] font-mono text-slate-500">
              <span className="text-accent-teal">Smart India Hackathon 2026</span>
              <span>•</span>
              <span>Problem Statement 26167</span>
              <span>•</span>
              <span className="text-white/70">ISRO Space Technology Theme</span>
            </div>
          </div>

          {/* Minimal Links */}
          <div className="flex items-center gap-6 text-slate-400 font-mono text-[11px]">
            <a
              href="#console"
              className="hover:text-accent-cyan transition-colors"
            >
              Interactive Console
            </a>
            <a
              href="#how-it-works"
              className="hover:text-accent-cyan transition-colors"
            >
              Architecture
            </a>
            <span className="text-white/20">|</span>
            <div className="flex items-center gap-1 text-slate-500">
              <Radio size={12} className="text-accent-cyan animate-pulse" />
              <span>UI Shell v0.1</span>
            </div>
          </div>
        </div>

        {/* Bottom copyright notice */}
        <div className="pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-[10px] font-mono text-slate-500">
          <div>
            Built for SIH 2026 • Sponsored by Indian Space Research Organisation (ISRO)
          </div>
          <div>
            FastAPI Backend Specification Ready
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
