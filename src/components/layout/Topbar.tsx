import { useLocation } from 'react-router-dom';
import { Menu, HelpCircle, FlaskConical, Sun, Moon } from 'lucide-react';
import { useApp } from '@/context/AppContext';

const pageTitles: Record<string, string> = {
  '/': 'Overview',
  '/new-analysis': 'New Analysis',
  '/model-selection': 'Model Selection & Configuration',
  '/history': 'Analysis History',
  '/explainability': 'Explainability Studio',
  '/model-performance': 'Model Performance',
  '/settings': 'Settings',
  '/docs': 'Documentation',
};

export default function Topbar() {
  const { setSidebarMobileOpen, isDemoMode, resolvedTheme, toggleTheme } = useApp();
  const location = useLocation();

  const pageTitle = pageTitles[location.pathname] ||
    (location.pathname.startsWith('/analysis/') ? 'Analysis Workspace' : 'NeuroLens AI');

  return (
    <header className="sticky top-0 z-20 bg-neuro-bg/80 backdrop-blur-md border-b border-neuro-border transition-colors">
      <div className="flex items-center justify-between h-14 px-4 lg:px-6">
        {/* Left: mobile menu + title */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarMobileOpen(true)}
            className="p-2 rounded-md hover:bg-neuro-surface-200 text-neuro-text-muted lg:hidden"
            aria-label="Open menu"
          >
            <Menu className="w-5 h-5" />
          </button>
          <div>
            <h2 className="text-sm font-semibold text-neuro-text-primary">{pageTitle}</h2>
          </div>
        </div>

        {/* Right: actions */}
        <div className="flex items-center gap-2">
          {isDemoMode && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20">
              <FlaskConical className="w-3.5 h-3.5 text-amber-500 dark:text-amber-400" />
              <span className="text-xs font-medium text-amber-600 dark:text-amber-400 hidden sm:inline">Demo Mode</span>
            </div>
          )}

          {/* Theme Quick Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-neuro-surface-200 text-neuro-text-muted hover:text-neuro-text-primary transition-colors"
            title={`Switch to ${resolvedTheme === 'dark' ? 'Light' : 'Dark'} mode`}
            aria-label="Toggle theme"
          >
            {resolvedTheme === 'dark' ? (
              <Sun className="w-4 h-4 text-amber-400" />
            ) : (
              <Moon className="w-4 h-4 text-neuro-accent" />
            )}
          </button>

          <button className="p-2 rounded-lg hover:bg-neuro-surface-200 text-neuro-text-muted hover:text-neuro-text-secondary transition-colors">
            <HelpCircle className="w-4 h-4" />
          </button>
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-neuro-accent to-neuro-cyan flex items-center justify-center text-xs font-bold text-white shadow-sm">
            R
          </div>
        </div>
      </div>
    </header>
  );
}
