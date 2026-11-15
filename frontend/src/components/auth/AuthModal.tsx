import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/context/AuthContext';
import {
  X,
  Lock,
  Mail,
  User as UserIcon,
  ShieldCheck,
  LogOut,
  Loader2,
  AlertCircle,
  Brain,
  CheckCircle2,
} from 'lucide-react';

export default function AuthModal() {
  const {
    user,
    isAuthenticated,
    showAuthModal,
    authModalMode,
    login,
    register,
    logout,
    openAuthModal,
    closeAuthModal,
  } = useAuth();

  const [mode, setMode] = useState<'login' | 'register'>(authModalMode);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('researcher');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (!showAuthModal) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      if (mode === 'login') {
        await login({ email, password });
      } else {
        await register({ name, email, password, role });
      }
    } catch (err: any) {
      setError(err?.message || 'Authentication failed. Please verify credentials.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        {/* Backdrop */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={closeAuthModal}
          className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        />

        {/* Modal Window */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          className="relative w-full max-w-md rounded-2xl bg-neuro-surface border border-neuro-border shadow-2xl p-6 overflow-hidden z-10"
        >
          {/* Header */}
          <div className="flex items-center justify-between pb-4 border-b border-neuro-border">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-neuro-accent/10 border border-neuro-accent/20 flex items-center justify-center text-neuro-accent">
                <Brain className="w-5 h-5 text-neuro-accent" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-neuro-text-primary">
                  {isAuthenticated ? 'Researcher Profile' : mode === 'login' ? 'Sign In' : 'Create Researcher Account'}
                </h3>
                <p className="text-xs text-neuro-text-muted">
                  {isAuthenticated
                    ? 'Authenticated Research Session'
                    : 'Access persistent analysis history & isolation'}
                </p>
              </div>
            </div>
            <button
              onClick={closeAuthModal}
              className="p-1.5 rounded-lg text-neuro-text-muted hover:text-neuro-text-primary hover:bg-neuro-surface-200 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Body */}
          {isAuthenticated && user ? (
            <div className="py-6 space-y-5">
              <div className="flex items-center gap-4 p-4 rounded-xl bg-neuro-surface-200/60 border border-neuro-border">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-neuro-accent to-neuro-cyan flex items-center justify-center text-base font-bold text-white shadow-md">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-neuro-text-primary">{user.name}</h4>
                  <p className="text-xs text-neuro-text-muted">{user.email}</p>
                  <span className="inline-block mt-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-neuro-accent/10 text-neuro-accent border border-neuro-accent/20 capitalize">
                    {user.role} Role
                  </span>
                </div>
              </div>

              <div className="space-y-2 text-xs text-neuro-text-secondary">
                <div className="flex justify-between py-1.5 border-b border-neuro-border/60">
                  <span>User Identifier:</span>
                  <span className="font-mono text-neuro-text-primary">{user.id}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-neuro-border/60">
                  <span>Data Security:</span>
                  <span className="text-emerald-500 flex items-center gap-1">
                    <ShieldCheck className="w-3.5 h-3.5" /> Isolated in MongoDB Atlas
                  </span>
                </div>
                <div className="flex justify-between py-1.5">
                  <span>Session Status:</span>
                  <span className="text-neuro-accent-bright font-medium flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Active JWT Access
                  </span>
                </div>
              </div>

              <button
                onClick={async () => {
                  await logout();
                  closeAuthModal();
                }}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20 text-xs font-semibold transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span>Sign Out of Session</span>
              </button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="py-5 space-y-4">
              {error && (
                <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-500">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}

              {mode === 'register' && (
                <div>
                  <label className="block text-xs font-medium text-neuro-text-secondary mb-1.5">
                    Full Name & Title
                  </label>
                  <div className="relative">
                    <UserIcon className="absolute left-3 top-2.5 w-4 h-4 text-neuro-text-muted" />
                    <input
                      type="text"
                      required
                      value={name}
                      onChange={e => setName(e.target.value)}
                      placeholder="Dr. Elena Rostova, MD"
                      className="w-full pl-9 pr-3 py-2 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs text-neuro-text-primary placeholder:text-neuro-text-muted focus:outline-none focus:border-neuro-accent"
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-xs font-medium text-neuro-text-secondary mb-1.5">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-2.5 w-4 h-4 text-neuro-text-muted" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="researcher@example.com"
                    className="w-full pl-9 pr-3 py-2 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs text-neuro-text-primary placeholder:text-neuro-text-muted focus:outline-none focus:border-neuro-accent"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-neuro-text-secondary mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-2.5 w-4 h-4 text-neuro-text-muted" />
                  <input
                    type="password"
                    required
                    minLength={8}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full pl-9 pr-3 py-2 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs text-neuro-text-primary placeholder:text-neuro-text-muted focus:outline-none focus:border-neuro-accent"
                  />
                </div>
              </div>

              {mode === 'register' && (
                <div>
                  <label className="block text-xs font-medium text-neuro-text-secondary mb-1.5">
                    Research Discipline
                  </label>
                  <select
                    value={role}
                    onChange={e => setRole(e.target.value)}
                    className="w-full px-3 py-2 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs text-neuro-text-primary focus:outline-none focus:border-neuro-accent"
                  >
                    <option value="researcher">Neuroradiological Researcher</option>
                    <option value="clinician">Clinical Neurologist</option>
                    <option value="ai_engineer">Machine Learning Scientist</option>
                  </select>
                </div>
              )}

              <button
                type="submit"
                disabled={submitting}
                className="w-full py-2.5 rounded-lg bg-gradient-to-r from-neuro-accent to-neuro-cyan text-white font-semibold text-xs shadow-md hover:shadow-neuro-accent/20 disabled:opacity-50 flex items-center justify-center gap-2 transition-all mt-2"
              >
                {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                <span>{mode === 'login' ? 'Sign In to Workspace' : 'Create Account'}</span>
              </button>

              <div className="pt-2 text-center text-xs text-neuro-text-muted">
                {mode === 'login' ? (
                  <span>
                    New to NeuroScan?{' '}
                    <button
                      type="button"
                      onClick={() => {
                        setError(null);
                        setMode('register');
                      }}
                      className="text-neuro-accent hover:underline font-semibold"
                    >
                      Create an account
                    </button>
                  </span>
                ) : (
                  <span>
                    Already have an account?{' '}
                    <button
                      type="button"
                      onClick={() => {
                        setError(null);
                        setMode('login');
                      }}
                      className="text-neuro-accent hover:underline font-semibold"
                    >
                      Sign In
                    </button>
                  </span>
                )}
              </div>
            </form>
          )}
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
