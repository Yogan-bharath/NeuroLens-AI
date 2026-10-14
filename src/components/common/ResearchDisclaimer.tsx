import { AlertTriangle } from 'lucide-react';
import { ShieldAlert } from 'lucide-react';

export function ResearchDisclaimer() {
  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-500/5 border border-amber-500/15">
      <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
      <p className="text-xs text-amber-300/80 leading-relaxed">
        <span className="font-medium text-amber-400">Research Use Only.</span>{' '}
        Model predictions are not medical diagnoses. Results should be reviewed by qualified professionals. Do not use for clinical decision-making.
      </p>
    </div>
  );
}

export function DemoModeBadge() {
  return (
    <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20">
      <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
      <span className="text-xs font-medium text-amber-400">Demo Mode — Mock Data</span>
    </div>
  );
}
