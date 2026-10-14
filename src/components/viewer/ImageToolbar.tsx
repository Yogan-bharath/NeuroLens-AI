import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Download,
  Eye,
  EyeOff,
  Layers,
} from 'lucide-react';

interface ImageToolbarProps {
  zoom: number;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
  onDownload: () => void;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  overlayOpacity: number;
  onChangeOverlayOpacity: (opacity: number) => void;
  hasOverlay: boolean;
  overlayLabel?: string;
}

export default function ImageToolbar({
  zoom,
  onZoomIn,
  onZoomOut,
  onReset,
  isFullscreen,
  onToggleFullscreen,
  onDownload,
  showOverlay,
  onToggleOverlay,
  overlayOpacity,
  onChangeOverlayOpacity,
  hasOverlay,
  overlayLabel = 'Overlay',
}: ImageToolbarProps) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 p-2 bg-neuro-surface-200/90 backdrop-blur-sm border border-neuro-border rounded-lg text-xs">
      {/* Zoom and view controls */}
      <div className="flex items-center gap-1">
        <button
          onClick={onZoomOut}
          className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-text-secondary hover:text-neuro-text-primary transition-colors"
          title="Zoom Out"
          aria-label="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <span className="w-12 text-center font-mono text-neuro-text-muted">
          {Math.round(zoom * 100)}%
        </span>
        <button
          onClick={onZoomIn}
          className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-text-secondary hover:text-neuro-text-primary transition-colors"
          title="Zoom In"
          aria-label="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={onReset}
          className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-text-secondary hover:text-neuro-text-primary transition-colors ml-1"
          title="Reset View"
          aria-label="Reset View"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>

      {/* Overlay controls if overlay available */}
      {hasOverlay && (
        <div className="flex items-center gap-2 px-2 py-1 bg-neuro-surface-300/60 rounded border border-neuro-border/60">
          <button
            onClick={onToggleOverlay}
            className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${
              showOverlay
                ? 'bg-neuro-accent-muted text-neuro-accent-bright'
                : 'text-neuro-text-muted hover:text-neuro-text-primary'
            }`}
            title={showOverlay ? 'Hide overlay' : 'Show overlay'}
          >
            {showOverlay ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span className="font-medium text-xs">{overlayLabel}</span>
          </button>

          {showOverlay && (
            <div className="flex items-center gap-1.5 pl-1 border-l border-neuro-border">
              <span className="text-[10px] text-neuro-text-muted">Opacity:</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={e => onChangeOverlayOpacity(parseFloat(e.target.value))}
                className="w-16 accent-neuro-accent h-1.5 bg-neuro-surface-400 rounded-lg cursor-pointer"
                title={`Overlay Opacity: ${Math.round(overlayOpacity * 100)}%`}
              />
              <span className="text-[10px] font-mono text-neuro-text-muted w-7">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Action buttons: fullscreen & download */}
      <div className="flex items-center gap-1">
        <button
          onClick={onDownload}
          className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-text-secondary hover:text-neuro-text-primary transition-colors"
          title="Download Current View"
          aria-label="Download Image"
        >
          <Download className="w-4 h-4" />
        </button>
        <button
          onClick={onToggleFullscreen}
          className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-text-secondary hover:text-neuro-text-primary transition-colors"
          title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          aria-label="Toggle Fullscreen"
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>
    </div>
  );
}
