import type { ConfusionMatrixData } from '@/types/model';

interface ConfusionMatrixProps {
  data: ConfusionMatrixData;
}

export default function ConfusionMatrix({ data }: ConfusionMatrixProps) {
  const { labels, matrix } = data;

  // Find max value in matrix for intensity scaling
  const maxVal = Math.max(...matrix.flatMap(row => row));

  return (
    <div className="space-y-4">
      <div className="overflow-x-auto">
        <table className="border-collapse mx-auto text-xs">
          <thead>
            <tr>
              <th className="p-2 text-center text-[10px] text-neuro-text-muted">
                Actual \ Predicted
              </th>
              {labels.map((lbl, idx) => (
                <th
                  key={idx}
                  className="p-2 text-center font-semibold text-neuro-text-primary text-[11px] min-w-[70px]"
                >
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((rowLabel, rIdx) => (
              <tr key={rIdx}>
                <td className="p-2 font-semibold text-neuro-text-primary text-[11px] whitespace-nowrap text-right pr-3">
                  {rowLabel}
                </td>
                {matrix[rIdx].map((val, cIdx) => {
                  const isDiagonal = rIdx === cIdx;
                  const ratio = val / maxVal;

                  // Highlighting diagonal vs off-diagonal
                  const bgStyle = isDiagonal
                    ? `rgba(20, 184, 166, ${0.15 + ratio * 0.75})`
                    : val > 0
                    ? `rgba(245, 158, 11, ${0.1 + (val / maxVal) * 0.5})`
                    : 'transparent';

                  return (
                    <td
                      key={cIdx}
                      style={{ backgroundColor: bgStyle }}
                      className="p-3 text-center border border-neuro-border font-mono font-medium text-neuro-text-primary rounded transition-transform hover:scale-105"
                      title={`Actual: ${rowLabel}, Predicted: ${labels[cIdx]}: ${val}`}
                    >
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-4 text-[11px] text-neuro-text-muted pt-2">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-neuro-accent/60" />
          <span>Diagonal (True Positives)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded bg-amber-500/30" />
          <span>Off-Diagonal (Misclassifications)</span>
        </div>
      </div>
    </div>
  );
}
