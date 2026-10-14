import { useState, useRef, useEffect, useCallback } from 'react';
import ImageToolbar from './ImageToolbar';
import { Columns, SplitSquareVertical } from 'lucide-react';

interface MRIImageViewerProps {
  originalImage: string;
  overlayImage?: string | null;
  overlayLabel?: string;
  title?: string;
  showComparisonOption?: boolean;
  defaultSideBySide?: boolean;
}

export default function MRIImageViewer({
  originalImage,
  overlayImage,
  overlayLabel = 'Heatmap Overlay',
  title = '2D Brain MRI Viewer',
  showComparisonOption = true,
  defaultSideBySide,
}: MRIImageViewerProps) {
  const isSegmentation = overlayLabel.toLowerCase().includes('segmentation');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [isSideBySide, setIsSideBySide] = useState(defaultSideBySide ?? isSegmentation);

  useEffect(() => {
    if (overlayLabel.toLowerCase().includes('segmentation')) {
      setIsSideBySide(true);
    }
  }, [overlayLabel]);

  const containerRef = useRef<HTMLDivElement>(null);

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 4));
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleToggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  useEffect(() => {
    const handleFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left click
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    setZoom(prev => Math.min(Math.max(prev + delta, 0.5), 4));
  };

  const handleDownload = useCallback(() => {
    // Render current composite to canvas and download
    const canvas = document.createElement('canvas');
    const img1 = new Image();
    img1.crossOrigin = 'anonymous';
    img1.onload = () => {
      canvas.width = img1.naturalWidth || 512;
      canvas.height = img1.naturalHeight || 512;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img1, 0, 0, canvas.width, canvas.height);

      if (overlayImage && showOverlay) {
        const img2 = new Image();
        img2.crossOrigin = 'anonymous';
        img2.onload = () => {
          ctx.globalAlpha = overlayOpacity;
          ctx.drawImage(img2, 0, 0, canvas.width, canvas.height);
          const link = document.createElement('a');
          link.download = `neurolens_mri_composite_${Date.now()}.png`;
          link.href = canvas.toDataURL('image/png');
          link.click();
        };
        img2.src = overlayImage;
      } else {
        const link = document.createElement('a');
        link.download = `neurolens_mri_${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      }
    };
    img1.src = originalImage;
  }, [originalImage, overlayImage, showOverlay, overlayOpacity]);

  return (
    <div
      ref={containerRef}
      className={`flex flex-col bg-neuro-surface border border-neuro-border rounded-xl overflow-hidden shadow-neuro-sm ${
        isFullscreen ? 'h-screen w-screen p-4' : 'h-[520px] lg:h-[600px]'
      }`}
    >
      {/* Viewer Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neuro-border bg-neuro-surface-200/50">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-neuro-accent" />
          <h3 className="text-sm font-semibold text-neuro-text-primary">{title}</h3>
          <span className="text-xs text-neuro-text-muted px-2 py-0.5 rounded bg-neuro-surface-300">
            2D Slice
          </span>
        </div>

        {showComparisonOption && overlayImage && (
          <button
            onClick={() => setIsSideBySide(prev => !prev)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              isSideBySide
                ? 'bg-neuro-accent text-white shadow-sm'
                : 'bg-neuro-surface-300 text-neuro-text-secondary hover:text-neuro-text-primary'
            }`}
          >
            {isSideBySide ? <Columns className="w-3.5 h-3.5" /> : <SplitSquareVertical className="w-3.5 h-3.5" />}
            <span>{isSideBySide ? 'Single View' : 'Side-by-Side'}</span>
          </button>
        )}
      </div>

      {/* Viewer Canvas Area - dedicated dark backdrop for DICOM MRI imaging contrast */}
      <div
        className="relative flex-1 bg-slate-950 overflow-hidden cursor-grab active:cursor-grabbing select-none"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        {/* Subtle grid pattern background */}
        <div
          className="absolute inset-0 opacity-10 pointer-events-none"
          style={{
            backgroundImage:
              'radial-gradient(circle, #334155 1px, transparent 1px)',
            backgroundSize: '20px 20px',
          }}
        />

        {isSideBySide && overlayImage ? (
          /* Side by side comparison */
          <div
            className="w-full h-full flex items-center justify-center gap-4 p-4 transition-transform duration-75"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
            }}
          >
            {/* Left: Original */}
            <div className="flex-1 max-w-[48%] flex flex-col items-center">
              <span className="text-xs font-medium text-neuro-text-muted mb-2 px-2 py-0.5 rounded bg-neuro-surface-300/80">
                Original Scan
              </span>
              <div className="relative rounded-lg overflow-hidden border border-neuro-border bg-black shadow-lg aspect-square w-full max-h-[420px] flex items-center justify-center">
                <img
                  src={originalImage}
                  alt="Original Brain MRI"
                  className="w-full h-full object-contain pointer-events-none"
                  draggable={false}
                />
              </div>
            </div>

            {/* Right: Overlay */}
            <div className="flex-1 max-w-[48%] flex flex-col items-center">
              <span className="text-xs font-medium text-neuro-accent-bright mb-2 px-2 py-0.5 rounded bg-neuro-accent-muted">
                {overlayLabel}
              </span>
              <div className="relative rounded-lg overflow-hidden border border-neuro-border bg-black shadow-lg aspect-square w-full max-h-[420px] flex items-center justify-center">
                <img
                  src={originalImage}
                  alt="Background MRI"
                  className="w-full h-full object-contain pointer-events-none"
                  draggable={false}
                />
                {showOverlay && (
                  <img
                    src={overlayImage}
                    alt={overlayLabel}
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen"
                    style={{ opacity: overlayOpacity }}
                    draggable={false}
                  />
                )}
              </div>
            </div>
          </div>
        ) : (
          /* Single unified overlay view */
          <div
            className="w-full h-full flex items-center justify-center transition-transform duration-75"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
            }}
          >
            <div className="relative rounded-lg overflow-hidden border border-neuro-border bg-black shadow-xl max-w-[90%] max-h-[90%] aspect-square flex items-center justify-center">
              <img
                src={originalImage}
                alt="Brain MRI Scan"
                className="w-full h-full object-contain pointer-events-none"
                draggable={false}
              />
              {overlayImage && showOverlay && (
                <img
                  src={overlayImage}
                  alt={overlayLabel}
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen"
                  style={{ opacity: overlayOpacity }}
                  draggable={false}
                />
              )}
            </div>
          </div>
        )}

        {/* Floating label indicators */}
        <div className="absolute top-3 left-3 pointer-events-none flex items-center gap-2">
          <span className="text-[11px] font-mono px-2 py-1 rounded bg-neuro-surface/80 backdrop-blur-sm border border-neuro-border text-neuro-text-secondary">
            {overlayImage && showOverlay ? overlayLabel : 'Original Scan'}
          </span>
        </div>

        {/* Pan instruction tooltip */}
        <div className="absolute bottom-3 right-3 pointer-events-none text-[10px] text-neuro-text-muted bg-neuro-surface/80 backdrop-blur-sm px-2 py-1 rounded border border-neuro-border/60">
          Scroll to zoom · Drag to pan
        </div>
      </div>

      {/* Floating Toolbar at the bottom */}
      <div className="p-3 border-t border-neuro-border bg-neuro-surface">
        <ImageToolbar
          zoom={zoom}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onReset={handleReset}
          isFullscreen={isFullscreen}
          onToggleFullscreen={handleToggleFullscreen}
          onDownload={handleDownload}
          showOverlay={showOverlay}
          onToggleOverlay={() => setShowOverlay(prev => !prev)}
          overlayOpacity={overlayOpacity}
          onChangeOverlayOpacity={setOverlayOpacity}
          hasOverlay={!!overlayImage}
          overlayLabel={overlayLabel}
        />
      </div>
    </div>
  );
}
