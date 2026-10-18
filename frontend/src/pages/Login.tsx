import React, { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Globe2, Lock, Mail } from 'lucide-react';
import { Button } from '../components/ui/Button';

interface LoginProps {
  mode?: 'login' | 'signup';
}

const Login: React.FC<LoginProps> = ({ mode = 'login' }) => {
  const navigate = useNavigate();
  const [pending, setPending] = useState(false);
  const isLogin = mode === 'login';

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    setPending(true);
    // No auth backend exists yet — this only opens the workspace.
    setTimeout(() => navigate('/console'), 400);
  };

  return (
    <div className="relative flex min-h-screen w-full items-center justify-center overflow-hidden bg-ground p-6">
      <div className="radar-grid pointer-events-none absolute inset-0" aria-hidden />
      <div
        className="pointer-events-none absolute left-1/2 top-1/3 h-[36rem] w-[36rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent/5 blur-[140px]"
        aria-hidden
      />

      <div className="relative w-full max-w-sm">
        <div className="mb-7 text-center">
          <span className="mx-auto mb-4 flex h-11 w-11 items-center justify-center rounded-xl border border-accent/25 bg-accent/10 text-accent">
            <Globe2 size={21} />
          </span>
          <h1 className="text-xl font-semibold tracking-tight text-ink">
            {isLogin ? 'Sign in to Diachron AI' : 'Create an account'}
          </h1>
          <p className="mt-1.5 text-[13px] text-ink-muted">
            Semantic retrieval and multi-temporal change intelligence.
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="space-y-3.5 rounded-panel border border-line bg-surface p-5"
        >
          <label className="block">
            <span className="label-caps text-ink-faint">Email</span>
            <span className="relative mt-1.5 flex items-center">
              <Mail size={15} className="absolute left-3 text-ink-faint" aria-hidden />
              <input
                type="email"
                required
                autoComplete="email"
                placeholder="name@agency.gov"
                className="h-10 w-full rounded-lg border border-line bg-surface-2 pl-9 pr-3 text-[13.5px] text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/50"
              />
            </span>
          </label>

          <label className="block">
            <span className="label-caps text-ink-faint">Password</span>
            <span className="relative mt-1.5 flex items-center">
              <Lock size={15} className="absolute left-3 text-ink-faint" aria-hidden />
              <input
                type="password"
                required
                autoComplete={isLogin ? 'current-password' : 'new-password'}
                placeholder="••••••••"
                className="h-10 w-full rounded-lg border border-line bg-surface-2 pl-9 pr-3 text-[13.5px] text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/50"
              />
            </span>
          </label>

          <Button type="submit" className="w-full" isLoading={pending}>
            {isLogin ? 'Sign in' : 'Create account'}
            {!pending && <ArrowRight size={15} />}
          </Button>
        </form>

        <p className="mt-4 text-center text-[12.5px] text-ink-faint">
          {isLogin ? 'No account yet?' : 'Already registered?'}{' '}
          <button
            type="button"
            onClick={() => navigate(isLogin ? '/signup' : '/login')}
            className="cursor-pointer text-accent transition-colors hover:text-accent/80"
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </p>

        <p className="mt-6 text-center font-mono text-[10px] leading-relaxed text-ink-faint">
          Demo only — these fields are not checked against any auth service.
        </p>
      </div>
    </div>
  );
};

export default Login;
