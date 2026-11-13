interface ConfidenceIndicatorProps {
  confidence: number;
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}

export default function ConfidenceIndicator({
  confidence,
  label = 'Model Confidence',
  size = 'md',
}: ConfidenceIndicatorProps) {
  const percentage = Math.round(confidence * 100);

  // SVG circular progress calculation
  const radius = size === 'sm' ? 24 : size === 'lg' ? 44 : 32;
  const stroke = size === 'sm' ? 4 : size === 'lg' ? 6 : 5;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (confidence * circumference);

  const dimension = radius * 2;

  // Color mapping based on score
  const color =
    percentage >= 80 ? '#14b8a6' : percentage >= 60 ? '#06b6d4' : percentage >= 40 ? '#f59e0b' : '#ef4444';

  return (
    <div className="flex items-center gap-3">
      <div className="relative flex items-center justify-center" style={{ width: dimension, height: dimension }}>
        <svg height={dimension} width={dimension} className="transform -rotate-90">
          <circle
            stroke="#1e293b"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          <circle
            stroke={color}
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${circumference} ${circumference}`}
            style={{ strokeDashoffset, transition: 'stroke-dashoffset 0.8s ease' }}
            strokeLinecap="round"
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>
        <div className="absolute text-center">
          <span className={`font-bold font-mono text-neuro-text-primary ${size === 'sm' ? 'text-xs' : size === 'lg' ? 'text-lg' : 'text-sm'}`}>
            {percentage}%
          </span>
        </div>
      </div>

      <div>
        <p className="text-xs font-medium text-neuro-text-primary">{label}</p>
        <p className="text-[11px] text-neuro-text-muted">
          {percentage >= 80 ? 'High probability match' : percentage >= 50 ? 'Moderate match' : 'Low confidence'}
        </p>
      </div>
    </div>
  );
}
