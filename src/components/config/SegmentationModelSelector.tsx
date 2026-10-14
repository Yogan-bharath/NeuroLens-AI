import type { SegmentationModel } from '@/types/modelRegistry';
import SegmentationModelCard from './SegmentationModelCard';

interface SegmentationModelSelectorProps {
  models: SegmentationModel[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}

export default function SegmentationModelSelector({
  models,
  selectedModelId,
  onSelectModel,
}: SegmentationModelSelectorProps) {
  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-neuro-text-primary uppercase tracking-wider text-xs">
          Step 2 (B): Segmentation Model
        </h3>
        <p className="text-xs text-neuro-text-muted mt-0.5">
          Select the encoder-decoder architecture for volumetric lesion and edema pixel-level segmentation.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {models.map(model => (
          <SegmentationModelCard
            key={model.id}
            model={model}
            isSelected={selectedModelId === model.id}
            onSelect={onSelectModel}
          />
        ))}
      </div>
    </div>
  );
}
