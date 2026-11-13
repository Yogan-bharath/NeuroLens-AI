import { AlertTriangle, X } from 'lucide-react';

interface CompatibilityWarningProps {
  message: string;
  onDismiss?: () => void;
}

export default function CompatibilityWarning({ message, onDismiss }: CompatibilityWarningProps) {
  return (
    <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 animate-fade-in">
      <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
      <div className="flex-1 leading-relaxed">
        <strong className="text-amber-400 font-semibold">Compatibility Note: </strong>
        {message}
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="p-1 rounded hover:bg-amber-500/20 text-amber-400 transition-colors"
          title="Dismiss warning"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}
