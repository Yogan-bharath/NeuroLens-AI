import type { XaiMethod, ClassificationModel } from '@/types/modelRegistry';
import XaiMethodCard from './XaiMethodCard';
import { xaiMethodsRegistry } from '@/services/modelRegistry';

interface XaiMethodSelectorProps {
  selectedMethodId: string;
  onSelectMethod: (methodId: string) => void;
  selectedModel: ClassificationModel | undefined;
}

export default function XaiMethodSelector({
  selectedMethodId,
  onSelectMethod,
  selectedModel,
}: XaiMethodSelectorProps) {
  const supportedMethods = selectedModel?.supportedXaiMethods || [];
  const modelName = selectedModel?.name || 'Selected Model';

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-neuro-text-primary uppercase tracking-wider text-xs">
          Step 3: Classification Explainable AI (XAI) Method
        </h3>
        <p className="text-xs text-neuro-text-muted mt-0.5">
          Select attribution algorithm compatible with <strong className="text-neuro-accent-bright">{modelName}</strong>.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {xaiMethodsRegistry.map(method => {
          const isCompatible = supportedMethods.includes(method.id);

          return (
            <XaiMethodCard
              key={method.id}
              method={method}
              isSelected={selectedMethodId === method.id}
              isCompatible={isCompatible}
              onSelect={onSelectMethod}
              activeModelName={modelName}
            />
          );
        })}
      </div>
    </div>
  );
}
