import { type LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export default function EmptyState({ icon: Icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
      <div className="w-14 h-14 rounded-xl bg-neuro-surface-200 flex items-center justify-center mb-4">
        <Icon className="w-7 h-7 text-neuro-text-muted" />
      </div>
      <h3 className="text-base font-medium text-neuro-text-primary mb-1">{title}</h3>
      <p className="text-sm text-neuro-text-muted max-w-sm">{description}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="neuro-btn-primary mt-4"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
