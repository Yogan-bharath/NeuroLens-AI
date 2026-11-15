import React, { useState, useEffect, useRef } from 'react';
import { AlertCircle, Loader2, Sliders, Layers } from 'lucide-react';

export interface SegExplainStats {
  min?: number;
  max?: number;
  mean?: number;
  hot_fraction?: number;
  heat_mass_in_mask?: number;
  mask_area_fraction?: number;
  degenerate?: boolean;
}

export interface SegExplainViewerProps {
  originalImage?: string;
  heatmapUrl?: string;
  maskUrl?: string;
  featureMapShape?: string;
  isAvailable?: boolean;
  unavailableReason?: string;
  blendMode?: 'screen' | 'overlay' | 'normal';
  opacity?: number;
  stats?: SegExplainStats;
  isLoading?: boolean;
}

export const SegExplainViewer: React.FC<SegExplainViewerProps> = ({
  originalImage,
  heatmapUrl,
  maskUrl,
  featureMapShape = '256 × 256 × 256',
  isAvailable = true,
  unavailableReason,
  blendMode = 'screen',
  opacity = 0.65,
  stats,
  isLoading = false,
}) => {
  const [showOutline, setShowOutline] = useState<boolean>(true);
  const [outlineDataUrl, setOutlineDataUrl] = useState<string | null>(null);
  const [outlineLoaded, setOutlineLoaded] = useState<boolean>(false);

  // Compute thin boundary outline from mask image
  useEffect(() => {
    if (!maskUrl) {
      setOutlineDataUrl(null);
      return;
    }

    let isMounted = true;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (!isMounted) return;
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      if (!w || !h) return;

      const srcCanvas = document.createElement('canvas');
      srcCanvas.width = w;
      srcCanvas.height = h;
      const sctx = srcCanvas.getContext('2d');
      if (!sctx) return;
      sctx.drawImage(img, 0, 0);

      const srcData = sctx.getImageData(0, 0, w, h);
      const s = srcData.data;

      const outCanvas = document.createElement('canvas');
      outCanvas.width = w;
      outCanvas.height = h;
      const octx = outCanvas.getContext('2d');
      if (!octx) return;
      const outImgData = octx.createImageData(w, h);
      const d = outImgData.data;

      // Extract 1-2px boundary contour: amber #F59E0B
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = (y * w + x) * 4;
          const isTumor = s[idx] > 100 && s[idx + 1] < 120; // Red mask pixel
          if (isTumor) {
            // Check 4-connectivity for border
            const up = ((y - 1) * w + x) * 4;
            const down = ((y + 1) * w + x) * 4;
            const left = (y * w + (x - 1)) * 4;
            const right = (y * w + (x + 1)) * 4;

            const isBorder =
              s[up] < 50 || s[down] < 50 || s[left] < 50 || s[right] < 50;

            if (isBorder) {
              // Amber #F59E0B
              d[idx] = 245;
              d[idx + 1] = 158;
              d[idx + 2] = 11;
              d[idx + 3] = 255;
            }
          }
        }
      }

      octx.putImageData(outImgData, 0, 0);
      setOutlineDataUrl(outCanvas.toDataURL('image/png'));
      setOutlineLoaded(true);
    };

    img.onerror = () => {
      setOutlineDataUrl(null);
    };

    img.src = maskUrl;

    return () => {
      isMounted = false;
    };
  }, [maskUrl]);

  const getBlendClass = () => {
    if (blendMode === 'screen') return 'mix-blend-screen';
    if (blendMode === 'overlay') return 'mix-blend-overlay';
    return 'mix-blend-normal';
  };

  return (
    <div className="space-y-3">
      {/* Layer Label & Thin Outline Toggle */}
      <div className="flex items-center justify-between flex-wrap gap-2 text-xs">
        <div className="flex items-center gap-1.5 bg-neuro-surface-300 p-0.5 rounded border border-neuro-border">
          <span className="text-[10px] text-neuro-text-muted px-1.5 font-semibold">LAYER:</span>
          <span className="px-2 py-0.5 rounded text-[11px] bg-neuro-accent text-neuro-bg font-bold shadow-xs">
            Fine (up_sampling2d_2, 256×256)
          </span>
        </div>

        <label className="flex items-center gap-1.5 cursor-pointer text-[11px] text-neuro-text-secondary hover:text-white select-none">
          <input
            type="checkbox"
            checked={showOutline}
            onChange={(e) => setShowOutline(e.target.checked)}
            className="rounded border-neuro-border bg-neuro-surface-300 text-amber-500 focus:ring-0"
          />
          <span className="flex items-center gap-1">
            <span className="inline-block w-2.5 h-2.5 rounded-full border border-amber-400 bg-amber-400/20" />
            Tumor Outline
          </span>
        </label>
      </div>

      {/* Main Heatmap Viewport */}
      <div className="relative aspect-square max-h-[400px] w-full rounded-lg bg-black overflow-hidden border border-neuro-border flex items-center justify-center">
        {!heatmapUrl ? (
          <div className="p-6 rounded-lg bg-black/60 border border-neuro-border/60 text-center space-y-2 max-w-[85%]">
            <AlertCircle className="w-8 h-8 text-amber-400 mx-auto opacity-80" />
            <p className="text-xs font-semibold text-white">
              No explanation available for this layer
            </p>
            <p className="text-[11px] text-neuro-text-muted">
              {unavailableReason ||
                'Activation gradients across this layer are degenerate (near-zero or constant). The model did not produce a differentiable signal for attribution.'}
            </p>
          </div>
        ) : (
          <>
            {/* Base MRI Image */}
            {originalImage && (
              <img
                src={originalImage}
                alt="Base Brain MRI"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
              />
            )}

            {/* Seg-Grad-CAM Smooth Heatmap PNG (NO filled mask) */}
            <img
              src={heatmapUrl}
              alt="Seg-Grad-CAM Heatmap"
              className={`absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity ${getBlendClass()}`}
              style={{ opacity }}
            />

            {/* Optional Thin Mask Outline Contour (NO fill) */}
            {showOutline && outlineDataUrl && (
              <img
                src={outlineDataUrl}
                alt="Predicted Tumor Boundary Outline"
                className="absolute inset-0 w-full h-full object-contain pointer-events-none opacity-90"
              />
            )}

            {/* HUD Overlay Indicators */}
            <div className="absolute top-2 left-2 pointer-events-none font-mono text-[10px] text-neuro-accent bg-black/75 px-2 py-1 rounded border border-white/10 backdrop-blur-xs">
              <div className="font-semibold text-white">SEG-GRAD-CAM EXPLAIN</div>
              <div className="text-neuro-text-muted">
                LAYER: up_sampling2d_2 · MAP: {featureMapShape}
              </div>
            </div>

            {stats && typeof stats.heat_mass_in_mask === 'number' && (
              <div className="absolute top-2 right-2 pointer-events-none font-mono text-[10px] text-right text-neuro-text-muted bg-black/75 px-2 py-1 rounded border border-white/10 backdrop-blur-xs">
                <div className="text-white font-semibold">
                  HEAT IN MASK: {(stats.heat_mass_in_mask * 100).toFixed(1)}%
                </div>
                <div>HOT PIXELS: {((stats.hot_fraction || 0) * 100).toFixed(1)}%</div>
              </div>
            )}

            <div className="absolute bottom-2 left-2 pointer-events-none font-mono text-[10px] text-neuro-text-muted bg-black/75 px-2 py-0.5 rounded border border-white/10">
              BLEND: {blendMode.toUpperCase()} ({Math.round(opacity * 100)}%)
            </div>
          </>
        )}
        {isLoading && (
          <div
            role="status"
            aria-live="polite"
            className="absolute inset-0 z-20 flex items-center justify-center bg-black/75 backdrop-blur-sm"
          >
            <div className="rounded-lg border border-neuro-border bg-neuro-surface-200/95 px-6 py-5 text-center shadow-xl">
              <Loader2 className="mx-auto h-8 w-8 animate-spin text-neuro-accent" />
              <p className="mt-3 text-sm font-semibold text-neuro-text-primary">
                Generating explanation…
              </p>
              <p className="mt-1 text-xs text-neuro-text-muted">
                The model is recalculating the selected layer. This may take a little while.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Jet Colormap Legend Bar */}
      {Boolean(heatmapUrl) && (
        <div className="space-y-1 pt-1">
          <div className="h-2 w-full rounded-full bg-gradient-to-r from-blue-700 via-cyan-400 via-yellow-400 to-red-600 shadow-inner border border-white/10" />
          <div className="flex justify-between items-center font-mono text-[10px] text-neuro-text-muted">
            <span>0.0 (Suppressed)</span>
            <span className="text-neuro-text-secondary">Saliency Attribution</span>
            <span>1.0 (Peak Saliency)</span>
          </div>
        </div>
      )}

      {/* Caption & Disclaimers */}
      <p className="text-[11px] text-neuro-text-muted leading-relaxed">
        Shows features that influenced total predicted tumor probability using the higher-resolution decoder layer.
        Grad-CAM is not constrained
        to the predicted mask, may highlight areas outside it, and is not a tumor boundary.
      </p>

      {typeof stats?.heat_mass_in_mask === 'number' && stats.heat_mass_in_mask < 0.01 && (
        <p className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/20 px-2.5 py-2 rounded">
          This layer’s saliency has little overlap with the predicted tumor region. Try the Fine layer
          for more spatial detail; its attribution may also extend outside the mask.
        </p>
      )}

      <div className="text-[10px] text-amber-400/90 font-mono bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded">
        Research/educational use only. Not clinically validated.
      </div>
    </div>
  );
};

