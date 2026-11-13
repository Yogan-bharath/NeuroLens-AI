import { cn } from '@/lib/utils';

interface LoadingStateProps {
  text?: string;
  className?: string;
}

export function LoadingSpinner({ text, className }: LoadingStateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center py-12', className)}>
      <div className="relative w-10 h-10">
        <div className="absolute inset-0 rounded-full border-2 border-neuro-surface-300" />
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-neuro-accent animate-spin" />
      </div>
      {text && <p className="mt-3 text-sm text-neuro-text-muted">{text}</p>}
    </div>
  );
}

export function SkeletonLine({ className }: { className?: string }) {
  return (
    <div className={cn('h-4 bg-neuro-surface-200 rounded animate-pulse', className)} />
  );
}

export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn('neuro-card space-y-3', className)}>
      <SkeletonLine className="w-1/3 h-3" />
      <SkeletonLine className="w-full h-5" />
      <SkeletonLine className="w-2/3 h-3" />
    </div>
  );
}
