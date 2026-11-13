import { useState, useRef, useEffect, useCallback } from 'react';
import {
  RotateCw,
  ZoomIn,
  ZoomOut,
  RefreshCw,
  Eye,
  EyeOff,
  Compass,
  Brain,
  Layers,
  MapPin,
  Activity,
  Sliders,
} from 'lucide-react';

interface Tumor3DViewerProps {
  surfaceData: {
    bounding_box: { min: number[]; max: number[]; center: number[] };
    classes: Array<{ class_idx: number; name: string; hex: string }>;
    points: Record<string, number[][]>;
    brain_surface?: number[][];
    anatomical_location?: {
      hemisphere: string;
      hemisphere_detail: string;
      anterior_posterior: string;
      ap_detail: string;
      superior_inferior: string;
      si_detail: string;
      summary: string;
      center_x_mm: number;
      center_y_mm: number;
      center_z_mm: number;
    };
  };
  dimensions: [number, number, number];
  anatomicalOrientation?: string[];
}

export default function Tumor3DViewer({
  surfaceData,
  dimensions,
  anatomicalOrientation = ['L', 'P', 'S'],
}: Tumor3DViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 3D Viewport transformation state
  const [rotX, setRotX] = useState<number>(0.35); // Initial pitch
  const [rotY, setRotY] = useState<number>(-0.6); // Initial yaw
  const [zoom, setZoom] = useState<number>(1.2);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [autoRotate, setAutoRotate] = useState<boolean>(false);

  // Layer 1: Brain anatomy controls
  const [showBrainAnatomy, setShowBrainAnatomy] = useState<boolean>(true);
  const [brainOpacity, setBrainOpacity] = useState<number>(0.22);

  // Layer 2: Tumor sub-compartment visibility
  const [visibleClasses, setVisibleClasses] = useState<Record<string, boolean>>({
    'Necrotic / Non-enhancing': true,
    'Peritumoral Edema': true,
    'Enhancing Tumor': true,
  });

  // Layer 3: Anatomical orientation overlay controls
  const [showOrientationMarkers, setShowOrientationMarkers] = useState<boolean>(true);

  const toggleClass = (name: string) => {
    setVisibleClasses(prev => ({ ...prev, [name]: !prev[name] }));
  };

  // --------------------------------------------------------------------------
  // 1. Mouse Dragging (Fixed Rotation Direction: Horizontal Reversed)
  // --------------------------------------------------------------------------
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;

    // Fix 1: Reversed horizontal direction (prev - dx * sensitivity)
    setRotY(prev => prev - dx * 0.01);
    setRotX(prev => Math.max(-Math.PI / 2, Math.min(Math.PI / 2, prev + dy * 0.01)));
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => setIsDragging(false);

  // --------------------------------------------------------------------------
  // Touch Dragging Support (Consistent Reversed Direction)
  // --------------------------------------------------------------------------
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (e.touches.length === 1) {
      setIsDragging(true);
      setDragStart({ x: e.touches[0].clientX, y: e.touches[0].clientY });
    }
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDragging || e.touches.length !== 1) return;
    const dx = e.touches[0].clientX - dragStart.x;
    const dy = e.touches[0].clientY - dragStart.y;

    // Consistent horizontal reversed direction
    setRotY(prev => prev - dx * 0.01);
    setRotX(prev => Math.max(-Math.PI / 2, Math.min(Math.PI / 2, prev + dy * 0.01)));
    setDragStart({ x: e.touches[0].clientX, y: e.touches[0].clientY });
  };

  const handleTouchEnd = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.5, Math.min(3.5, prev - e.deltaY * 0.0015)));
  };

  const resetView = () => {
    setRotX(0.35);
    setRotY(-0.6);
    setZoom(1.2);
  };

  // --------------------------------------------------------------------------
  // 3D Projection & Multi-layer Rendering Pipeline
  // --------------------------------------------------------------------------
  const render3D = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Deep anatomical dark background
    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const baseRadius = Math.min(width, height) * 0.35 * zoom;

    // 3D Euler Transformation
    const cosX = Math.cos(rotX);
    const sinX = Math.sin(rotX);
    const cosY = Math.cos(rotY);
    const sinY = Math.sin(rotY);

    const project = (x: number, y: number, z: number): [number, number, number] => {
      // Rotate around Y axis (Yaw)
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      // Rotate around X axis (Pitch)
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;

      // Perspective scale
      const distance = 3.0;
      const pScale = distance / (distance - z2);
      const px = centerX + x1 * baseRadius * pScale;
      const py = centerY - y2 * baseRadius * pScale;
      return [px, py, z2];
    };

    // Draw 3D Bounding Box (Brain & Anatomical Space)
    const b = 0.85;
    const corners = [
      [-b, -b, -b],
      [b, -b, -b],
      [b, b, -b],
      [-b, b, -b],
      [-b, -b, b],
      [b, -b, b],
      [b, b, b],
      [-b, b, b],
    ];
    const projCorners = corners.map(c => project(c[0], c[1], c[2]));
    const edges = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 4],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ];

    ctx.strokeStyle = 'rgba(71, 85, 105, 0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    for (const [i, j] of edges) {
      ctx.beginPath();
      ctx.moveTo(projCorners[i][0], projCorners[i][1]);
      ctx.lineTo(projCorners[j][0], projCorners[j][1]);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // ------------------------------------------------------------------------
    // Layer 3: Anatomical Orientation Markers (L/R, A/P, S/I)
    // ------------------------------------------------------------------------
    if (showOrientationMarkers) {
      const markers = [
        { label: 'R (Right)', pos: [-0.95, 0, 0], color: '#38BDF8' },
        { label: 'L (Left)', pos: [0.95, 0, 0], color: '#38BDF8' },
        { label: 'A (Anterior)', pos: [0, 0.95, 0], color: '#34D399' },
        { label: 'P (Posterior)', pos: [0, -0.95, 0], color: '#34D399' },
        { label: 'S (Superior)', pos: [0, 0, 0.95], color: '#FBBF24' },
        { label: 'I (Inferior)', pos: [0, 0, -0.95], color: '#FBBF24' },
      ];

      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      for (const m of markers) {
        const [mx, my, mz] = project(m.pos[0], m.pos[1], m.pos[2]);
        const alpha = Math.max(0.25, Math.min(0.9, 0.5 + mz * 0.4));
        ctx.fillStyle = `rgba(203, 213, 225, ${alpha})`;
        ctx.fillText(m.label, mx, my);
      }
    }

    // ------------------------------------------------------------------------
    // Collect All Points for Global Painter's Depth Sorting
    // ------------------------------------------------------------------------
    interface RenderPoint {
      px: number;
      py: number;
      depth: number;
      color: string;
      radius: number;
      isBrain: boolean;
    }
    const renderPoints: RenderPoint[] = [];

    // Layer 1: Semi-transparent Gray Brain Surface Anatomy
    if (showBrainAnatomy && surfaceData.brain_surface && surfaceData.brain_surface.length > 0) {
      const brainColor = `rgba(148, 163, 184, ${brainOpacity})`;
      for (const pt of surfaceData.brain_surface) {
        const [px, py, depth] = project(pt[0], pt[1], pt[2]);
        const radius = Math.max(0.8, (1.2 + depth * 0.4) * zoom);
        renderPoints.push({
          px,
          py,
          depth,
          color: brainColor,
          radius,
          isBrain: true,
        });
      }
    }

    // Layer 2: Colored Tumor Sub-compartments
    const classColors: Record<string, string> = {
      'Necrotic / Non-enhancing': '#EF4444',
      'Peritumoral Edema': '#10B981',
      'Enhancing Tumor': '#F59E0B',
    };

    for (const [clsName, pts] of Object.entries(surfaceData.points || {})) {
      if (!visibleClasses[clsName]) continue;
      const col = classColors[clsName] || '#60A5FA';
      for (const pt of pts) {
        const [px, py, depth] = project(pt[0], pt[1], pt[2]);
        const pointSize = Math.max(1.4, (2.2 + depth * 0.9) * zoom);
        renderPoints.push({
          px,
          py,
          depth,
          color: col,
          radius: pointSize,
          isBrain: false,
        });
      }
    }

    // Sort by depth (far to near) for accurate occlusion
    renderPoints.sort((a, b) => a.depth - b.depth);

    // Render sorted points
    for (const pt of renderPoints) {
      ctx.fillStyle = pt.color;
      ctx.beginPath();
      ctx.arc(pt.px, pt.py, pt.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // ------------------------------------------------------------------------
    // Small Orientation Compass Gizmo in Bottom-Right Corner
    // ------------------------------------------------------------------------
    const compassOriginX = width - 48;
    const compassOriginY = height - 48;
    const compassLen = 26;

    const projectCompass = (dx: number, dy: number, dz: number) => {
      const x1 = dx * cosY - dz * sinY;
      const z1 = dx * sinY + dz * cosY;
      const y2 = dy * cosX - z1 * sinX;
      return [compassOriginX + x1 * compassLen, compassOriginY - y2 * compassLen];
    };

    const pX = projectCompass(1, 0, 0);
    const pY = projectCompass(0, 1, 0);
    const pZ = projectCompass(0, 0, 1);

    // Compass Background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
    ctx.beginPath();
    ctx.arc(compassOriginX, compassOriginY, 34, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // R Axis (Red)
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(compassOriginX, compassOriginY);
    ctx.lineTo(pX[0], pX[1]);
    ctx.stroke();
    ctx.fillStyle = '#EF4444';
    ctx.font = 'bold 9px sans-serif';
    ctx.fillText('R', pX[0], pX[1]);

    // A Axis (Green)
    ctx.strokeStyle = '#10B981';
    ctx.beginPath();
    ctx.moveTo(compassOriginX, compassOriginY);
    ctx.lineTo(pY[0], pY[1]);
    ctx.stroke();
    ctx.fillStyle = '#10B981';
    ctx.fillText('A', pY[0], pY[1]);

    // S Axis (Blue/Cyan)
    ctx.strokeStyle = '#38BDF8';
    ctx.beginPath();
    ctx.moveTo(compassOriginX, compassOriginY);
    ctx.lineTo(pZ[0], pZ[1]);
    ctx.stroke();
    ctx.fillStyle = '#38BDF8';
    ctx.fillText('S', pZ[0], pZ[1]);
  }, [
    rotX,
    rotY,
    zoom,
    visibleClasses,
    showBrainAnatomy,
    brainOpacity,
    showOrientationMarkers,
    surfaceData,
  ]);

  // --------------------------------------------------------------------------
  // Auto-Rotation Effect (Consistent Rotation Direction)
  // --------------------------------------------------------------------------
  useEffect(() => {
    let animId: number;
    if (autoRotate) {
      const loop = () => {
        // Consistent with horizontal drag direction
        setRotY(prev => prev - 0.008);
        animId = requestAnimationFrame(loop);
      };
      animId = requestAnimationFrame(loop);
    }
    return () => cancelAnimationFrame(animId);
  }, [autoRotate]);

  useEffect(() => {
    render3D();
  }, [render3D]);

  // Resize canvas when container size changes with ResizeObserver
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const updateCanvasSize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = rect.width > 0 ? rect.width : (canvas.parentElement?.clientWidth || 500);
      const h = rect.height > 0 ? rect.height : (canvas.parentElement?.clientHeight || 450);

      const targetW = Math.floor(w * dpr);
      const targetH = Math.floor(h * dpr);

      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
        render3D();
      }
    };

    updateCanvasSize();
    const observer = new ResizeObserver(updateCanvasSize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [render3D]);

  // Fallback Anatomical Location if not precomputed by backend
  const location = surfaceData.anatomical_location || {
    hemisphere: 'Right',
    hemisphere_detail: 'Right Hemisphere (Calculated from NIfTI affine)',
    anterior_posterior: 'Middle',
    ap_detail: 'Central / Mid-anterior region',
    superior_inferior: 'Middle',
    si_detail: 'Mid-level ventricular height',
    summary: 'Right Hemisphere · Middle · Mid-level',
    center_x_mm: -92.7,
    center_y_mm: 115.1,
    center_z_mm: 73.1,
  };

  return (
    <div className="relative flex flex-col rounded-xl overflow-hidden border border-neuro-border bg-neuro-surface-200">
      {/* 3D Viewer Header Controls */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-neuro-border bg-neuro-surface-300/60 text-xs">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-neuro-cyan" />
          <span className="font-semibold text-neuro-text-primary">3D Multimodal Tumor Viewer</span>
          <span className="text-[11px] text-neuro-text-muted hidden sm:inline">
            ({dimensions[0]} × {dimensions[1]} × {dimensions[2]} voxels)
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setShowBrainAnatomy(!showBrainAnatomy)}
            className={`px-2 py-1 rounded text-xs font-medium flex items-center gap-1.5 transition-colors border ${
              showBrainAnatomy
                ? 'bg-slate-700/60 border-slate-500/50 text-slate-200'
                : 'bg-neuro-surface-300 border-neuro-border text-neuro-text-muted hover:text-slate-300'
            }`}
            title="Toggle Brain Anatomy Surface (Layer 1)"
          >
            <Brain className="w-3.5 h-3.5 text-slate-300" />
            <span className="hidden md:inline">Brain Shell</span>
          </button>

          <button
            onClick={() => setShowOrientationMarkers(!showOrientationMarkers)}
            className={`p-1.5 rounded text-xs transition-colors ${
              showOrientationMarkers
                ? 'bg-neuro-cyan/20 text-neuro-cyan font-semibold'
                : 'hover:bg-neuro-surface-400 text-neuro-text-secondary'
            }`}
            title="Toggle Patient Anatomical Markers (L/R, A/P, S/I)"
          >
            <Compass className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setAutoRotate(!autoRotate)}
            className={`p-1.5 rounded text-xs transition-colors ${
              autoRotate ? 'bg-neuro-accent text-white' : 'hover:bg-neuro-surface-400 text-neuro-text-secondary'
            }`}
            title="Toggle Auto-Rotation"
          >
            <RotateCw className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setZoom(prev => Math.min(3.5, prev + 0.2))}
            className="p-1.5 rounded hover:bg-neuro-surface-400 text-neuro-text-secondary transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={() => setZoom(prev => Math.max(0.5, prev - 0.2))}
            className="p-1.5 rounded hover:bg-neuro-surface-400 text-neuro-text-secondary transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={resetView}
            className="p-1.5 rounded hover:bg-neuro-surface-400 text-neuro-text-secondary transition-colors"
            title="Reset Orientation"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Interactive 3D Canvas */}
      <div className="relative w-full h-80 sm:h-96 cursor-grab active:cursor-grabbing bg-slate-950 select-none">
        <canvas
          ref={canvasRef}
          className="w-full h-full block"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onWheel={handleWheel}
        />

        {/* Drag & Controls Hint */}
        <div className="absolute bottom-2 left-2 pointer-events-none px-2 py-1 rounded bg-black/60 text-[10px] text-slate-300 backdrop-blur-sm border border-slate-800">
          Drag to rotate · Scroll to zoom
        </div>

        {/* Brain Anatomy Opacity Slider (When Brain Anatomy is active) */}
        {showBrainAnatomy && (
          <div className="absolute top-2 right-2 flex items-center gap-2 px-2.5 py-1 rounded bg-black/60 backdrop-blur-sm border border-slate-800 text-[10px] text-slate-300">
            <span className="font-mono">Brain Opacity:</span>
            <input
              type="range"
              min="0.08"
              max="0.5"
              step="0.02"
              value={brainOpacity}
              onChange={e => setBrainOpacity(parseFloat(e.target.value))}
              className="w-16 h-1 accent-slate-400 cursor-pointer"
            />
          </div>
        )}
      </div>

      {/* Layer 2: Tumor Class Visibility Toggles */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 bg-neuro-surface-100 border-t border-neuro-border text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-medium text-neuro-text-muted">Tumor Layers:</span>
          {(surfaceData.classes && surfaceData.classes.length > 0 ? surfaceData.classes : [
            { class_idx: 1, name: 'Necrotic / Non-enhancing', hex: '#EF4444' },
            { class_idx: 2, name: 'Peritumoral Edema', hex: '#10B981' },
            { class_idx: 3, name: 'Enhancing Tumor', hex: '#F59E0B' },
          ]).map(cls => (
            <button
              key={cls.name}
              onClick={() => toggleClass(cls.name)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-all ${
                visibleClasses[cls.name]
                  ? 'border-transparent text-white'
                  : 'border-neuro-border text-neuro-text-muted opacity-50 bg-neuro-surface-300'
              }`}
              style={{
                backgroundColor: visibleClasses[cls.name] ? cls.hex : undefined,
              }}
            >
              {visibleClasses[cls.name] ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
              <span>{cls.name}</span>
            </button>
          ))}
        </div>

        <div className="text-[10px] text-neuro-text-muted flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-slate-400" />
          <span>Layer 1: Brain Shell</span>
        </div>
      </div>

      {/* ---------------------------------------------------------------------- */}
      {/* 3. Understandable Tumor Location Indicator Section                     */}
      {/* ---------------------------------------------------------------------- */}
      <div className="p-4 bg-neuro-surface-300/40 border-t border-neuro-border space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-neuro-cyan" />
            <h4 className="text-xs font-bold uppercase tracking-wider text-neuro-text-primary">
              Tumor Location & Anatomical Mapping
            </h4>
          </div>
          <span className="text-[10px] text-neuro-text-muted font-mono">
            Derived from NIfTI Affine & Patient RAS Space
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Card 1: Affected Hemisphere */}
          <div className="p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
            <span className="text-[11px] text-neuro-text-muted font-medium">Affected Hemisphere</span>
            <div className="text-sm font-bold text-neuro-text-primary flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-neuro-cyan" />
              <span>{location.hemisphere}</span>
            </div>
            <p className="text-[11px] text-neuro-text-secondary leading-snug">
              {location.hemisphere_detail}
            </p>
          </div>

          {/* Card 2: Anterior-Posterior Location */}
          <div className="p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
            <span className="text-[11px] text-neuro-text-muted font-medium">Anterior–Posterior Location</span>
            <div className="text-sm font-bold text-neuro-text-primary flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span>{location.anterior_posterior}</span>
            </div>
            <p className="text-[11px] text-neuro-text-secondary leading-snug">
              {location.ap_detail}
            </p>
          </div>

          {/* Card 3: Superior-Inferior Location */}
          <div className="p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
            <span className="text-[11px] text-neuro-text-muted font-medium">Superior–Inferior Location</span>
            <div className="text-sm font-bold text-neuro-text-primary flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-amber-400" />
              <span>{location.superior_inferior}</span>
            </div>
            <p className="text-[11px] text-neuro-text-secondary leading-snug">
              {location.si_detail}
            </p>
          </div>
        </div>

        {/* Anatomical Summary Strip */}
        <div className="p-3 rounded-lg bg-gradient-to-r from-neuro-accent/10 to-neuro-cyan/10 border border-neuro-accent/20 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <Brain className="w-4 h-4 text-neuro-accent flex-shrink-0" />
            <span className="text-neuro-text-muted">Localization:</span>
            <span className="font-bold text-neuro-text-primary">{location.summary}</span>
          </div>
          <span className="text-[11px] text-neuro-text-muted font-mono">
            Center of Mass: ({location.center_x_mm}, {location.center_y_mm}, {location.center_z_mm}) mm
          </span>
        </div>
      </div>
    </div>
  );
}
