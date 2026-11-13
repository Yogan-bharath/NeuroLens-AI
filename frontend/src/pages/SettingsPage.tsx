import { useState } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { useApp, type ThemeMode } from '@/context/AppContext';
import {
  Sliders,
  Eye,
  Sparkles,
  Trash2,
  Check,
  Moon,
  Sun,
  Monitor,
  Palette,
} from 'lucide-react';

export default function SettingsPage() {
  const { theme, resolvedTheme, setTheme } = useApp();
  const [defaultMethod, setDefaultMethod] = useState<'gradcam' | 'gradcam_pp' | 'segmentation'>('gradcam');
  const [defaultOpacity, setDefaultOpacity] = useState(0.7);
  const [showSideBySideByDefault, setShowSideBySideByDefault] = useState(false);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleClearSession = () => {
    if (confirm('Clear local demo session state?')) {
      window.location.reload();
    }
  };

  const themeOptions: {
    id: ThemeMode;
    title: string;
    subtitle: string;
    description: string;
    icon: typeof Moon;
    palette: string[];
  }[] = [
    {
      id: 'dark',
      title: 'Midnight Slate',
      subtitle: 'Dark Theme',
      description: 'Comfortable deep navy-slate canvas (#0F172A) with solid elevated cards and electric teal accents. Ideal for radiology reading rooms.',
      icon: Moon,
      palette: ['#0f172a', '#18212f', '#14b8a6', '#06b6d4', '#f1f5f9'],
    },
    {
      id: 'light',
      title: 'Clinical Slate',
      subtitle: 'Light Theme (Gather AI)',
      description: 'Crisp medical canvas (#F8FAFC) with pure white cards (#FFFFFF), refined slate borders, and high-contrast text for bright clinical offices.',
      icon: Sun,
      palette: ['#f8fafc', '#ffffff', '#0d9488', '#0891b2', '#0f172a'],
    },
    {
      id: 'system',
      title: 'System Auto',
      subtitle: 'OS Preference',
      description: 'Automatically synchronizes between Midnight Slate and Clinical Slate based on your operating system preferences.',
      icon: Monitor,
      palette: ['#0f172a', '#f8fafc', '#14b8a6', '#0d9488'],
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-3xl space-y-8"
    >
      <PageHeader
        title="Settings & Preferences"
        subtitle="Configure viewer defaults, appearance themes, and explainability parameters."
      />

      {/* Theme preferences */}
      <div className="neuro-card space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neuro-accent/15 flex items-center justify-center">
              <Palette className="w-4 h-4 text-neuro-accent" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-neuro-text-primary">
                Interface Theme
              </h3>
              <p className="text-xs text-neuro-text-muted">
                Active: {resolvedTheme === 'dark' ? 'Midnight Slate (Dark)' : 'Clinical Slate (Light)'}
                {theme === 'system' ? ' • Auto-synced to OS' : ''}
              </p>
            </div>
          </div>
          <span className="neuro-badge-accent">
            {resolvedTheme === 'dark' ? 'Dark Mode' : 'Light Mode'}
          </span>
        </div>

        {/* Theme cards grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {themeOptions.map(opt => {
            const isSelected = theme === opt.id;
            const Icon = opt.icon;

            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => setTheme(opt.id)}
                className={`relative flex flex-col justify-between p-4 rounded-xl border text-left transition-all duration-200 group ${
                  isSelected
                    ? 'border-neuro-accent bg-neuro-surface shadow-neuro-sm ring-1 ring-neuro-accent/30'
                    : 'border-neuro-border bg-neuro-surface-200/50 hover:bg-neuro-surface hover:border-neuro-border-hover'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                      <div
                        className={`p-2 rounded-lg transition-colors ${
                          isSelected
                            ? 'bg-neuro-accent text-white shadow-sm'
                            : 'bg-neuro-surface-200 text-neuro-text-secondary group-hover:text-neuro-text-primary'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <span className="font-semibold text-xs text-neuro-text-primary block">
                          {opt.title}
                        </span>
                        <span className="text-[10px] text-neuro-text-muted">
                          {opt.subtitle}
                        </span>
                      </div>
                    </div>

                    <div
                      className={`w-5 h-5 rounded-full border flex items-center justify-center transition-all ${
                        isSelected
                          ? 'border-neuro-accent bg-neuro-accent text-white'
                          : 'border-neuro-border bg-neuro-surface-300'
                      }`}
                    >
                      {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                    </div>
                  </div>

                  <p className="text-xs text-neuro-text-muted leading-relaxed mb-4">
                    {opt.description}
                  </p>
                </div>

                <div className="pt-3 border-t border-neuro-border flex items-center justify-between">
                  <span className="text-[10px] font-medium text-neuro-text-muted uppercase tracking-wider">
                    {opt.id === 'system' ? 'Adaptive' : 'Palette'}
                  </span>
                  <div className="flex items-center gap-1.5">
                    {opt.palette.map((color, i) => (
                      <span
                        key={i}
                        className="w-3.5 h-3.5 rounded-full ring-1 ring-black/10 dark:ring-white/10 shadow-sm"
                        style={{ backgroundColor: color }}
                        title={color}
                      />
                    ))}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Viewer defaults */}
      <div className="neuro-card space-y-5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-neuro-cyan/10 flex items-center justify-center">
            <Eye className="w-4 h-4 text-neuro-cyan" />
          </div>
          <h3 className="text-sm font-semibold text-neuro-text-primary">Viewer Preferences</h3>
        </div>

        <div className="neuro-divider" />

        <div className="space-y-5 text-xs">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-neuro-text-primary text-sm">Default Overlay Opacity</p>
              <p className="text-neuro-text-muted mt-0.5">Starting transparency level for heatmaps and segmentation</p>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={defaultOpacity}
                onChange={e => setDefaultOpacity(parseFloat(e.target.value))}
                className="w-28 accent-[rgb(var(--neuro-accent))] h-1.5 bg-neuro-surface-300 rounded-full cursor-pointer"
              />
              <span className="font-mono text-neuro-text-primary w-10 text-right font-semibold">
                {Math.round(defaultOpacity * 100)}%
              </span>
            </div>
          </div>

          <div className="neuro-divider" />

          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-neuro-text-primary text-sm">Default Comparison Mode</p>
              <p className="text-neuro-text-muted mt-0.5">Launch analysis workspace with dual side-by-side view</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={showSideBySideByDefault}
                onChange={e => setShowSideBySideByDefault(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-10 h-5 bg-neuro-surface-300 peer-focus:outline-none rounded-full peer-checked:bg-neuro-accent transition-colors duration-300 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all after:duration-300 peer-checked:after:translate-x-5 shadow-inner" />
            </label>
          </div>
        </div>
      </div>

      {/* Default explanation method */}
      <div className="neuro-card space-y-5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-neuro-accent/10 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-neuro-accent" />
          </div>
          <h3 className="text-sm font-semibold text-neuro-text-primary">Explainability Defaults</h3>
        </div>

        <div className="neuro-divider" />

        <div className="space-y-4 text-xs">
          <p className="text-sm text-neuro-text-secondary">
            Select the default attribution method loaded upon analysis completion:
          </p>
          <div className="grid grid-cols-3 gap-3">
            {[
              { id: 'gradcam', label: 'Grad-CAM', desc: 'Standard Saliency' },
              { id: 'gradcam_pp', label: 'Grad-CAM++', desc: 'High-order derivatives' },
              { id: 'segmentation', label: 'Segmentation', desc: 'Predicted mask' },
            ].map(m => (
              <button
                key={m.id}
                onClick={() => setDefaultMethod(m.id as any)}
                className={`p-4 rounded-xl border text-left transition-all duration-200 ${
                  defaultMethod === m.id
                    ? 'border-neuro-accent bg-neuro-accent/10 shadow-neuro-sm ring-1 ring-neuro-accent/30'
                    : 'border-neuro-border bg-neuro-surface-200/50 text-neuro-text-secondary hover:bg-neuro-surface hover:border-neuro-border-hover'
                }`}
              >
                <p className="font-semibold text-sm text-neuro-text-primary">{m.label}</p>
                <p className="text-xs text-neuro-text-muted mt-1">{m.desc}</p>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Session controls */}
      <div className="neuro-card space-y-5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-neuro-surface-300 flex items-center justify-center">
            <Sliders className="w-4 h-4 text-neuro-text-secondary" />
          </div>
          <h3 className="text-sm font-semibold text-neuro-text-primary">Session Controls</h3>
        </div>

        <div className="neuro-divider" />

        <div className="flex items-center justify-between text-xs">
          <div>
            <p className="font-medium text-neuro-text-primary text-sm">Reset Demo Session</p>
            <p className="text-neuro-text-muted mt-1">
              Clear temporary analyses and reset mock inferences back to initial state.
            </p>
          </div>
          <button
            onClick={handleClearSession}
            className="neuro-btn-secondary text-xs text-neuro-error hover:bg-neuro-error/10 border-neuro-error/20 hover:border-neuro-error/30"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Save action */}
      <div className="flex justify-end gap-3 pt-2">
        <button
          onClick={handleSave}
          className="neuro-btn-primary text-sm flex items-center gap-2.5 group"
        >
          {saved ? <Check className="w-4 h-4" /> : null}
          <span>{saved ? 'Preferences Saved' : 'Save Preferences'}</span>
          {!saved && (
            <span className="w-5 h-5 rounded-full bg-white/20 inline-flex items-center justify-center text-[11px] group-hover:translate-x-0.5 transition-transform">
              →
            </span>
          )}
        </button>
      </div>

      <ResearchDisclaimer />
    </motion.div>
  );
}
