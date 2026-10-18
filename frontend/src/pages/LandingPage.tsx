import React from 'react';
import { Link } from 'react-router-dom';
import { Globe2 } from 'lucide-react';
import Hero from '../components/features/Hero';
import HowItWorks from '../components/features/HowItWorks';
import Capabilities from '../components/features/Capabilities';
import TechStack from '../components/features/TechStack';
import Footer from '../components/features/Footer';

const NAV = [
  { href: '#how-it-works', label: 'How it works' },
  { href: '#capabilities', label: 'Capabilities' },
  { href: '#stack', label: 'Stack' },
];

/**
 * Marketing page. The live workspace used to be embedded here, which made the
 * landing page and the product the same screen; the console now lives at
 * /console and this page only points to it.
 */
const LandingPage: React.FC = () => (
  <div className="min-h-screen bg-ground text-ink selection:bg-accent/30">
    <header className="fixed inset-x-0 top-0 z-50 border-b border-line/60 bg-ground/70 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-5">
        <Link to="/" className="flex items-center gap-2 text-[15px] font-semibold tracking-tight">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-accent/25 bg-accent/10 text-accent">
            <Globe2 size={15} />
          </span>
          Diachron <span className="text-accent">AI</span>
        </Link>

        <nav className="hidden items-center gap-6 md:flex" aria-label="Sections">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="text-[13px] text-ink-muted transition-colors hover:text-ink"
            >
              {item.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            to="/login"
            className="hidden rounded-lg px-3 py-1.5 text-[13px] text-ink-muted transition-colors hover:text-ink sm:block"
          >
            Sign in
          </Link>
          <Link
            to="/console"
            className="rounded-lg bg-accent px-3.5 py-1.5 text-[13px] font-medium text-space-black transition-colors hover:bg-accent/90"
          >
            Open workspace
          </Link>
        </div>
      </div>
    </header>

    <Hero />
    <HowItWorks />
    <Capabilities />
    <TechStack />
    <Footer />
  </div>
);

export default LandingPage;
