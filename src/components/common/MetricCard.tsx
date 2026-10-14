import { type LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  trend?: string;
  trendUp?: boolean;
  accent?: 'teal' | 'cyan' | 'blue' | 'amber';
  className?: string;
}

const accentColors = {
  teal: 'from-neuro-accent/20 to-transparent text-neuro-accent',
  cyan: 'from-neuro-cyan/20 to-transparent text-neuro-cyan',
  blue: 'from-neuro-blue/20 to-transparent text-neuro-blue',
  amber: 'from-amber-500/20 to-transparent text-amber-400',
};

export default function MetricCard({ label, value, icon: Icon, trend, trendUp, accent = 'teal', className }: MetricCardProps) {
  return (
    <div className={cn('neuro-card-hover relative overflow-hidden', className)}>
      <div className={cn('absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl rounded-bl-full opacity-50', accentColors[accent])} />
      <div className="relative">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-neuro-text-muted uppercase tracking-wide">{label}</span>
          <Icon className={cn('w-4 h-4', accentColors[accent].split(' ').pop())} />
        </div>
        <div className="text-2xl font-bold text-neuro-text-primary">{value}</div>
        {trend && (
          <p className={cn('text-xs mt-1', trendUp ? 'text-neuro-success' : 'text-neuro-text-muted')}>
            {trend}
          </p>
        )}
      </div>
    </div>
  );
}
