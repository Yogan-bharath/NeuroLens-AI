import { useState } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import {
  Sliders,
  Eye,
  Sparkles,
  Trash2,
  Check,
  Moon,
} from 'lucide-react';

export default function SettingsPage() {
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

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="max-w-3xl space-y-6"
    >
      <PageHeader
        title="Settings & Workspace Preferences"
        subtitle="Configure viewer defaults, XAI parameters, and local session options."
      />

      {/* Theme preferences */}
      <div className="neuro-card space-y-4">
        <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
          <Moon className="w-4 h-4 text-neuro-accent" />
          <h3 className="text-sm font-semibold text-neuro-text-primary">Interface Theme</h3>
        </div>
        <div className="flex items-center justify-between text-xs">
          <div>
            <p className="font-medium text-neuro-text-primary">Medical Imaging Dark Mode</p>
            <p className="text-neuro-text-muted mt-0.5">
              NeuroLens AI is designed dark-mode-first to optimize contrast and dynamic range for radiological inspection.
            </p>
          </div>
          <span className="neuro-badge-accent">Active (Dark Navy)</span>
        </div>
      </div>

      {/* Viewer defaults */}
      <div className="neuro-card space-y-4">
        <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
          <Eye className="w-4 h-4 text-neuro-cyan" />
          <h3 className="text-sm font-semibold text-neuro-text-primary">Viewer Preferences</h3>
        </div>

        <div className="space-y-4 text-xs">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-neuro-text-primary">Default Overlay Opacity</p>
              <p className="text-neuro-text-muted">Starting transparency level for heatmaps and segmentation</p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={defaultOpacity}
                onChange={e => setDefaultOpacity(parseFloat(e.target.value))}
                className="w-24 accent-neuro-accent h-1.5 bg-neuro-surface-300 rounded"
              />
              <span className="font-mono text-neuro-text-primary w-8 text-right">
                {Math.round(defaultOpacity * 100)}%
              </span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-3 border-t border-neuro-border">
            <div>
              <p className="font-medium text-neuro-text-primary">Default Comparison Mode</p>
              <p className="text-neuro-text-muted">Launch analysis workspace with dual side-by-side view</p>
            </div>
            <input
              type="checkbox"
              checked={showSideBySideByDefault}
              onChange={e => setShowSideBySideByDefault(e.target.checked)}
              className="accent-neuro-accent w-4 h-4 rounded cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Default explanation method */}
      <div className="neuro-card space-y-4">
        <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
          <Sparkles className="w-4 h-4 text-neuro-blue" />
          <h3 className="text-sm font-semibold text-neuro-text-primary">Explainability Defaults</h3>
        </div>

        <div className="space-y-3 text-xs">
          <p className="text-neuro-text-secondary">
            Select the default attribution method loaded upon analysis completion:
          </p>
          <div className="grid grid-cols-3 gap-2">
            {[
              { id: 'gradcam', label: 'Grad-CAM', desc: 'Standard Saliency' },
              { id: 'gradcam_pp', label: 'Grad-CAM++', desc: 'High-order derivatives' },
              { id: 'segmentation', label: 'Segmentation', desc: 'Predicted mask' },
            ].map(m => (
              <button
                key={m.id}
                onClick={() => setDefaultMethod(m.id as any)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  defaultMethod === m.id
                    ? 'border-neuro-accent bg-neuro-accent-muted text-neuro-accent-bright'
                    : 'border-neuro-border bg-neuro-surface-200 text-neuro-text-secondary hover:border-neuro-border-hover'
                }`}
              >
                <p className="font-semibold">{m.label}</p>
                <p className="text-[10px] text-neuro-text-muted mt-0.5">{m.desc}</p>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Session controls */}
      <div className="neuro-card space-y-4">
        <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
          <Sliders className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-neuro-text-primary">Session Controls</h3>
        </div>

        <div className="flex items-center justify-between text-xs">
          <div>
            <p className="font-medium text-neuro-text-primary">Reset Demo Session</p>
            <p className="text-neuro-text-muted mt-0.5">
              Clear temporary analyses and reset mock inferences back to initial state.
            </p>
          </div>
          <button
            onClick={handleClearSession}
            className="neuro-btn-secondary text-xs text-neuro-error hover:bg-neuro-error/15"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Reset Demo</span>
          </button>
        </div>
      </div>

      {/* Save action */}
      <div className="flex justify-end gap-3 pt-2">
        <button onClick={handleSave} className="neuro-btn-primary text-xs">
          {saved ? <Check className="w-4 h-4" /> : null}
          <span>{saved ? 'Preferences Saved' : 'Save Preferences'}</span>
        </button>
      </div>

      <ResearchDisclaimer />
    </motion.div>
  );
}
