import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { motion } from 'framer-motion';
import {
  Upload,
  X,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Brain,
  Layers,
  Check,
  ShieldCheck,
  Info,
  Sliders,
  FileCheck,
  AlertTriangle,
  RefreshCw,
  FolderUp,
  FolderSearch,
  ChevronDown,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { formatFileSize } from '@/lib/utils';
import {
  analyzeClassification2D,
  analyzeSegmentationVolume,
  validateSegmentationStudy,
  getSegmentationStatus,
  type SegmentationStatusResponse,
  type StudyValidationReport,
} from '@/services/api';
import { useConfiguration } from '@/context/ConfigurationContext';
import {
  classificationModelsRegistry,
  volumetricSegmentationModelsRegistry,
} from '@/services/modelRegistry';
import {
  inspectStudyFolder,
  getFilesFromDataTransferItems,
  type ModalityKey,
  type FolderInspectionResult,
} from '@/lib/modalityDetector';

// Supported formats
const ALLOWED_2D_TYPES = { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] };
const MAX_2D_SIZE = 15 * 1024 * 1024; // 15MB

type WorkflowMode = 'classification' | 'segmentation';

export default function NewAnalysisPage() {
  const navigate = useNavigate();
  const folderInputRef = useRef<HTMLInputElement>(null);

  // Workflow A: 2D Classification | Workflow B: 3D Segmentation
  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowMode>('classification');

  const {
    selectedClassificationModelId,
    selectedSegmentationModelId,
    setClassificationModel,
  } = useConfiguration();

  const availableClassificationModels = useMemo(
    () => classificationModelsRegistry.filter(m => m.status === 'available'),
    []
  );

  const activeClassificationModel = useMemo(
    () => classificationModelsRegistry.find(m => m.id === selectedClassificationModelId) || classificationModelsRegistry[0],
    [selectedClassificationModelId]
  );

  const activeSegmentationModel = useMemo(
    () => volumetricSegmentationModelsRegistry.find(m => m.id === selectedSegmentationModelId) || volumetricSegmentationModelsRegistry[0],
    [selectedSegmentationModelId]
  );

  // ==========================================
  // WORKFLOW A: 2D CLASSIFICATION STATE
  // ==========================================
  const [file2D, setFile2D] = useState<File | null>(null);
  const [preview2D, setPreview2D] = useState<string | null>(null);
  const [dimensions2D, setDimensions2D] = useState<{ w: number; h: number } | null>(null);
  const [useClahe, setUseClahe] = useState<boolean>(true);
  const [isAnalyzing2D, setIsAnalyzing2D] = useState<boolean>(false);
  const [error2D, setError2D] = useState<string | null>(null);

  // ==========================================
  // WORKFLOW B: 3D FOLDER-BASED SEGMENTATION STATE
  // ==========================================
  const [inspectionResult, setInspectionResult] = useState<FolderInspectionResult | null>(null);
  const [selectedModalityFiles, setSelectedModalityFiles] = useState<Record<ModalityKey, File | null>>({
    flair: null,
    t1: null,
    t1ce: null,
    t2: null,
    seg: null,
  });
  const [validationReport, setValidationReport] = useState<StudyValidationReport | null>(null);
  const [isValidatingStudy, setIsValidatingStudy] = useState<boolean>(false);
  const [isDragOverFolder, setIsDragOverFolder] = useState<boolean>(false);
  const [segmentationStatus, setSegmentationStatus] = useState<SegmentationStatusResponse | null>(null);
  const [isAnalyzing3D, setIsAnalyzing3D] = useState<boolean>(false);
  const [error3D, setError3D] = useState<string | null>(null);
  const [stageProgress3D, setStageProgress3D] = useState<string>('');

  const handleSelectWorkflow = (mode: WorkflowMode) => {
    setActiveWorkflow(mode);
    setError2D(null);
    setError3D(null);
  };

  // Fetch segmentation model status on mount
  useEffect(() => {
    let active = true;
    getSegmentationStatus()
      .then(st => {
        if (active) setSegmentationStatus(st);
      })
      .catch(() => {
        if (active) setSegmentationStatus(null);
      });
    return () => {
      active = false;
    };
  }, []);

  // ----------------------------------------------------
  // WORKFLOW A: 2D Dropzone Handlers
  // ----------------------------------------------------
  const onDrop2D = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    setError2D(null);
    if (rejectedFiles.length > 0) {
      const rej = rejectedFiles[0];
      const code = rej.errors[0]?.code;
      if (code === 'file-too-large') {
        setError2D('File exceeds the 15 MB maximum size limit.');
      } else if (code === 'file-invalid-type') {
        setError2D('Unsupported file type. Please upload a standard 2D MRI slice (PNG, JPG, or JPEG).');
      } else {
        setError2D(rej.errors[0]?.message || 'File upload failed.');
      }
      return;
    }

    if (acceptedFiles.length > 0) {
      const f = acceptedFiles[0];
      setFile2D(f);
      const url = URL.createObjectURL(f);
      setPreview2D(url);

      const reader = new FileReader();
      reader.onload = () => {
        const img = new Image();
        img.onload = () => setDimensions2D({ w: img.width, h: img.height });
        img.src = url;
      };
      reader.readAsDataURL(f);
    }
  }, []);

  const { getRootProps: getRoot2D, getInputProps: getInput2D, isDragActive: isDrag2D } = useDropzone({
    onDrop: onDrop2D,
    accept: ALLOWED_2D_TYPES,
    maxSize: MAX_2D_SIZE,
    multiple: false,
    disabled: isAnalyzing2D,
  });

  const handleRun2DClassification = async () => {
    if (!file2D || isAnalyzing2D) return;
    setError2D(null);
    setIsAnalyzing2D(true);
    try {
      const res = await analyzeClassification2D(
        file2D,
        selectedClassificationModelId || 'efficientnetv2b0_256',
        useClahe
      );
      if (res.analysis_id) {
        navigate(`/analysis/${res.analysis_id}/2d-results`);
      } else {
        throw new Error('Analysis completed without an analysis ID.');
      }
    } catch (err) {
      setError2D(err instanceof Error ? err.message : 'Classification failed.');
      setIsAnalyzing2D(false);
    }
  };

  // ----------------------------------------------------
  // WORKFLOW B: Folder Inspection & Modality Detection
  // ----------------------------------------------------
  const processFolderFiles = useCallback(async (files: File[]) => {
    setError3D(null);
    const inspection = inspectStudyFolder(files);

    if (inspection.totalNiftiCount === 0) {
      setError3D('No valid NIfTI files (.nii or .nii.gz) were detected in the selected folder.');
      setInspectionResult(null);
      setValidationReport(null);
      return;
    }

    setInspectionResult(inspection);

    const initialSelection: Record<ModalityKey, File | null> = {
      flair: inspection.modalities.flair.selectedFile,
      t1: inspection.modalities.t1.selectedFile,
      t1ce: inspection.modalities.t1ce.selectedFile,
      t2: inspection.modalities.t2.selectedFile,
      seg: inspection.modalities.seg.selectedFile,
    };
    setSelectedModalityFiles(initialSelection);
    setValidationReport(null);

    // If all 4 required modalities are uniquely detected without ambiguity, auto-trigger validation!
    if (inspection.allRequiredPresent && !inspection.hasAmbiguity) {
      runStudyValidation(initialSelection);
    }
  }, []);

  const handleFolderInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const filesArray = Array.from(e.target.files);
      processFolderFiles(filesArray);
    }
  };

  const handleFolderDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOverFolder(false);
    setError3D(null);

    const items = e.dataTransfer.items;
    let files: File[] = [];

    if (items && items.length > 0) {
      files = await getFilesFromDataTransferItems(items);
    } else if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      files = Array.from(e.dataTransfer.files);
    }

    if (files.length === 0) {
      setError3D('No files detected in the dropped directory.');
      return;
    }

    processFolderFiles(files);
  };

  const handleSelectAmbiguousFile = (modalityKey: ModalityKey, fileName: string) => {
    if (!inspectionResult) return;
    const candidates = inspectionResult.modalities[modalityKey].candidates;
    const chosen = candidates.find(f => f.name === fileName) || null;

    setSelectedModalityFiles(prev => {
      const updated = { ...prev, [modalityKey]: chosen };
      if (updated.flair && updated.t1 && updated.t1ce && updated.t2) {
        runStudyValidation(updated);
      }
      return updated;
    });
  };

  // Run spatial and integrity validation on the backend
  const runStudyValidation = async (filesMap: Record<ModalityKey, File | null>) => {
    if (!filesMap.flair || !filesMap.t1 || !filesMap.t1ce || !filesMap.t2) {
      return;
    }
    setIsValidatingStudy(true);
    setError3D(null);

    try {
      const formData = new FormData();
      formData.append('flair', filesMap.flair);
      formData.append('t1', filesMap.t1);
      formData.append('t1ce', filesMap.t1ce);
      formData.append('t2', filesMap.t2);
      if (filesMap.seg) {
        formData.append('seg', filesMap.seg);
      }

      const report = await validateSegmentationStudy(formData);
      setValidationReport(report);
      if (!report.is_valid && report.errors.length > 0) {
        setError3D(report.errors[0]);
      }
    } catch (err) {
      setError3D(err instanceof Error ? err.message : 'Study validation failed.');
      setValidationReport(null);
    } finally {
      setIsValidatingStudy(false);
    }
  };

  const handleRun3DSegmentation = async () => {
    const { flair, t1, t1ce, t2, seg } = selectedModalityFiles;
    if (!flair || !t1 || !t1ce || !t2) {
      setError3D('All four core MRI modalities (FLAIR, T1, T1CE, T2) are required.');
      return;
    }
    if (!validationReport?.is_valid) {
      setError3D('All spatial alignment and NIfTI integrity checks must pass before inference.');
      return;
    }

    setError3D(null);
    setIsAnalyzing3D(true);
    setStageProgress3D('Uploading verified 4 multimodal NIfTI volumes...');

    try {
      const formData = new FormData();
      formData.append('flair', flair);
      formData.append('t1', t1);
      formData.append('t1ce', t1ce);
      formData.append('t2', t2);
      if (seg) {
        formData.append('seg', seg);
      }
      formData.append('model_id', selectedSegmentationModelId || 'brats_3d_unet');

      setStageProgress3D('Preprocessing, non-zero normalization & sliding-window inference...');
      const res = await analyzeSegmentationVolume(formData);
      if (res.analysis_id) {
        navigate(`/analysis/${res.analysis_id}/3d-results`);
      } else {
        throw new Error('3D Segmentation completed without an analysis ID.');
      }
    } catch (err) {
      setError3D(err instanceof Error ? err.message : '3D Segmentation failed.');
      setIsAnalyzing3D(false);
      setStageProgress3D('');
    }
  };

  const isModelReady = Boolean(segmentationStatus?.segmentation_ready);
  const allModalitiesSelected = Boolean(
    selectedModalityFiles.flair &&
    selectedModalityFiles.t1 &&
    selectedModalityFiles.t1ce &&
    selectedModalityFiles.t2
  );
  const canRun3D = allModalitiesSelected && validationReport?.is_valid && !isValidatingStudy && !isAnalyzing3D;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-6 max-w-5xl mx-auto pb-12 text-neuro-text-primary"
    >
      <PageHeader
        title="MRI Analysis Hub"
        subtitle="Select your clinical analysis workflow, upload compatible MRI data, and execute genuine AI inference pipelines."
      />

      {/* ─── TWO DEDICATED WORKFLOW SELECTION CARDS ───────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Workflow A: 2D Classification */}
        <button
          type="button"
          onClick={() => handleSelectWorkflow('classification')}
          className={`p-5 rounded-xl border text-left transition-all space-y-3 cursor-pointer ${
            activeWorkflow === 'classification'
              ? 'bg-neuro-surface-200 border-neuro-accent shadow-lg shadow-neuro-accent/5 ring-1 ring-neuro-accent'
              : 'bg-neuro-surface-200/50 border-neuro-border hover:border-neuro-border/80'
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 flex-shrink-0">
              <Brain className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400 font-mono">Workflow A</span>
              <h2 className="text-sm font-bold text-white">2D Brain Tumor Classification</h2>
            </div>
          </div>
          <p className="text-xs text-neuro-text-secondary leading-relaxed">
            Four-class pathology (Glioma, Meningioma, Pituitary, No Tumor) from individual 2D MRI slices with Grad-CAM saliency heatmaps.
          </p>
          <div className="flex items-center gap-2 pt-1 font-mono text-[10px] text-neuro-text-muted">
            <span className="px-2 py-0.5 rounded bg-neuro-surface-300 text-emerald-400 font-semibold">PNG / JPG</span>
            <span>·</span>
            <span>{activeClassificationModel?.name || 'EfficientNetV2B0'}</span>
          </div>
        </button>

        {/* Workflow B: 3D Segmentation */}
        <button
          type="button"
          onClick={() => handleSelectWorkflow('segmentation')}
          className={`p-5 rounded-xl border text-left transition-all space-y-3 cursor-pointer ${
            activeWorkflow === 'segmentation'
              ? 'bg-neuro-surface-200 border-cyan-400 shadow-lg shadow-cyan-400/5 ring-1 ring-cyan-400'
              : 'bg-neuro-surface-200/50 border-neuro-border hover:border-neuro-border/80'
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 flex-shrink-0">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono">Workflow B</span>
              <h2 className="text-sm font-bold text-white">3D Multimodal Segmentation</h2>
            </div>
          </div>
          <p className="text-xs text-neuro-text-secondary leading-relaxed">
            Volumetric 3D subregion segmentation using complete MRI study folders with automatic FLAIR/T1/T1CE/T2 detection and spatial alignment.
          </p>
          <div className="flex items-center gap-2 pt-1 font-mono text-[10px] text-neuro-text-muted">
            <span className="px-2 py-0.5 rounded bg-neuro-surface-300 text-cyan-400 font-semibold">MRI Study Folder (.nii)</span>
            <span>·</span>
            <span>Attention CNN + ViT</span>
          </div>
        </button>
      </div>

      {/* ========================================================================= */}
      {/* WORKFLOW A: 2D CLASSIFICATION INPUT                                      */}
      {/* ========================================================================= */}
      {activeWorkflow === 'classification' && (
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 flex-shrink-0">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-xs font-bold text-white font-mono">
                    Target Model: <span className="text-emerald-400">{activeClassificationModel?.name}</span>
                  </p>
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    SELECTED
                  </span>
                </div>
                <p className="text-[11px] text-neuro-text-muted mt-0.5">
                  {activeClassificationModel?.architecture} · {activeClassificationModel?.version}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4 flex-wrap">
              <div className="flex items-center gap-1 bg-neuro-surface-300 p-1 rounded-lg border border-neuro-border text-[11px] font-mono">
                {availableClassificationModels.map(m => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setClassificationModel(m.id)}
                    className={`px-2 py-0.5 rounded transition-all ${
                      selectedClassificationModelId === m.id
                        ? 'bg-neuro-accent text-white font-bold shadow-sm'
                        : 'text-neuro-text-muted hover:text-white'
                    }`}
                  >
                    {m.name}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <label htmlFor="clahe-toggle" className="text-xs text-neuro-text-secondary flex items-center gap-1.5 cursor-pointer">
                  <span>CLAHE</span>
                </label>
                <input
                  id="clahe-toggle"
                  type="checkbox"
                  checked={useClahe}
                  onChange={e => setUseClahe(e.target.checked)}
                  className="w-4 h-4 rounded text-neuro-accent focus:ring-0 cursor-pointer"
                  title="Enable CLAHE Contrast Normalization"
                />
              </div>
            </div>
          </div>

          <div className="space-y-4">
            {!preview2D ? (
              <div
                {...getRoot2D()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
                  isDrag2D
                    ? 'border-neuro-accent bg-neuro-accent/5'
                    : 'border-neuro-border hover:border-neuro-accent/60 bg-neuro-surface-200/50'
                }`}
              >
                <input {...getInput2D()} />
                <div className="flex flex-col items-center gap-3">
                  <div className="w-12 h-12 rounded-xl bg-neuro-surface-300 flex items-center justify-center text-neuro-accent">
                    <Upload className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-white">
                      Drop 2D Brain MRI Scan Here or <span className="text-neuro-accent">Browse</span>
                    </p>
                    <p className="text-xs text-neuro-text-muted mt-1 font-mono">
                      Supports PNG, JPG, or JPEG · Maximum file size 15 MB
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="relative p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-col sm:flex-row items-center gap-4">
                <div className="w-24 h-24 rounded-lg bg-black overflow-hidden flex-shrink-0 border border-neuro-border">
                  <img src={preview2D} alt="2D Preview" className="w-full h-full object-contain" />
                </div>
                <div className="flex-1 space-y-1 text-xs">
                  <p className="font-semibold text-white truncate max-w-sm">{file2D?.name}</p>
                  <p className="text-neuro-text-muted font-mono">
                    Size: {file2D ? formatFileSize(file2D.size) : '0 B'} · Dimensions: {dimensions2D ? `${dimensions2D.w} × ${dimensions2D.h} px` : 'Detecting...'}
                  </p>
                  <div className="flex items-center gap-2 pt-1">
                    <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                      Ready for Inference
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setFile2D(null);
                    setPreview2D(null);
                    setDimensions2D(null);
                  }}
                  className="p-2 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-muted hover:text-red-400 transition-colors"
                  title="Remove image"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {error2D && (
              <div className="p-3.5 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-2.5 text-xs text-red-400">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{error2D}</span>
              </div>
            )}
          </div>

          <button
            type="button"
            disabled={!file2D || isAnalyzing2D}
            onClick={handleRun2DClassification}
            className={`w-full py-3 px-4 rounded-xl font-mono text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-md ${
              file2D && !isAnalyzing2D
                ? 'bg-neuro-accent hover:bg-neuro-accent-bright text-white cursor-pointer'
                : 'bg-neuro-surface-300 text-neuro-text-muted cursor-not-allowed opacity-60'
            }`}
          >
            {isAnalyzing2D ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Running 2D Classification & Grad-CAM...</span>
              </>
            ) : (
              <>
                <Brain className="w-4 h-4" />
                <span>Run 2D Classification Analysis</span>
              </>
            )}
          </button>

          <ResearchDisclaimer />
        </div>
      )}

      {/* ========================================================================= */}
      {/* WORKFLOW B: 3D MULTIMODAL FOLDER SEGMENTATION                            */}
      {/* ========================================================================= */}
      {activeWorkflow === 'segmentation' && (
        <div className="space-y-6">
          {/* Compact Selected Model Panel */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 flex-shrink-0">
                <Brain className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <p className="text-xs font-bold text-white font-mono">
                    Selected Segmentation Model: <span className="text-cyan-400">{activeSegmentationModel?.name}</span>
                  </p>
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                    VERIFIED
                  </span>
                </div>
                <p className="text-[11px] text-neuro-text-muted mt-0.5">
                  {activeSegmentationModel?.architecture} · {activeSegmentationModel?.version}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className={`px-2.5 py-1 rounded text-[11px] font-semibold border font-mono ${isModelReady ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30'}`}>
                {isModelReady ? 'Active in Memory' : 'Candidate Loaded'}
              </span>
              <Link
                to="/models"
                className="text-xs font-mono text-cyan-400 hover:text-cyan-300 underline underline-offset-2 whitespace-nowrap"
              >
                Model Details →
              </Link>
            </div>
          </div>

          {/* Hidden Directory File Input */}
          <input
            type="file"
            ref={folderInputRef}
            // @ts-ignore
            webkitdirectory=""
            directory=""
            mozdirectory=""
            multiple
            onChange={handleFolderInputChange}
            className="hidden"
          />

          {/* Folder Upload Dropzone Area */}
          <div
            onDragOver={e => {
              e.preventDefault();
              setIsDragOverFolder(true);
            }}
            onDragLeave={() => setIsDragOverFolder(false)}
            onDrop={handleFolderDrop}
            className={`border-2 border-dashed rounded-xl p-8 sm:p-10 text-center transition-all ${
              isDragOverFolder
                ? 'border-cyan-400 bg-cyan-500/5 scale-[0.99]'
                : inspectionResult
                ? 'border-neuro-border bg-neuro-surface-200/50'
                : 'border-neuro-border hover:border-cyan-400/60 bg-neuro-surface-200/50 cursor-pointer'
            }`}
            onClick={() => {
              if (!inspectionResult) {
                folderInputRef.current?.click();
              }
            }}
          >
            <div className="max-w-md mx-auto flex flex-col items-center gap-3">
              <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 shadow-sm">
                {inspectionResult ? (
                  <FolderSearch className="w-7 h-7" />
                ) : (
                  <FolderUp className="w-7 h-7" />
                )}
              </div>

              <div>
                <h3 className="text-base font-bold text-white">
                  {inspectionResult ? `Study Folder: ${inspectionResult.folderName}` : 'Select MRI Study Folder'}
                </h3>
                <p className="text-xs text-neuro-text-secondary mt-1">
                  {inspectionResult
                    ? `Scanned ${inspectionResult.totalNiftiCount} volumetric NIfTI file(s) across subdirectories.`
                    : 'Choose the folder containing your multimodal NIfTI MRI sequences.'}
                </p>
                <p className="text-[11px] font-mono text-neuro-text-muted mt-2">
                  Expected inputs: .nii · .nii.gz (FLAIR, T1, T1CE, T2)
                </p>
              </div>

              <div className="pt-2 flex items-center gap-3">
                <button
                  type="button"
                  onClick={e => {
                    e.stopPropagation();
                    folderInputRef.current?.click();
                  }}
                  className="px-4 py-2 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-mono font-semibold transition-all shadow-sm flex items-center gap-2 cursor-pointer"
                >
                  <FolderUp className="w-4 h-4" />
                  <span>{inspectionResult ? 'Change Study Folder' : 'Browse Folder'}</span>
                </button>

                {inspectionResult && (
                  <button
                    type="button"
                    onClick={e => {
                      e.stopPropagation();
                      setInspectionResult(null);
                      setSelectedModalityFiles({ flair: null, t1: null, t1ce: null, t2: null, seg: null });
                      setValidationReport(null);
                      setError3D(null);
                    }}
                    className="px-3 py-2 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-muted hover:text-white border border-neuro-border text-xs font-mono transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Automatic Modality Detection List */}
          {inspectionResult && (
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-neuro-border pb-2">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  <h4 className="text-xs font-bold uppercase tracking-wider font-mono text-white">
                    Detected MRI Modalities ({[selectedModalityFiles.flair, selectedModalityFiles.t1, selectedModalityFiles.t1ce, selectedModalityFiles.t2].filter(Boolean).length}/4 Required)
                  </h4>
                </div>
                {isValidatingStudy && (
                  <span className="flex items-center gap-1.5 text-xs text-cyan-400 font-mono">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Validating Headers & Affines...</span>
                  </span>
                )}
              </div>

              {/* Modality Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {(['flair', 't1', 't1ce', 't2', 'seg'] as ModalityKey[]).map(key => {
                  const mod = inspectionResult.modalities[key];
                  const selectedFile = selectedModalityFiles[key];
                  const hasSelection = Boolean(selectedFile);
                  const isAmbiguous = mod.isAmbiguous;
                  const isMissing = mod.required && !hasSelection;

                  // Skip optional SEG if neither detected nor present
                  if (key === 'seg' && mod.candidates.length === 0) return null;

                  return (
                    <div
                      key={key}
                      className={`p-4 rounded-xl border bg-neuro-surface-200 transition-all space-y-2.5 ${
                        hasSelection
                          ? 'border-neuro-border'
                          : isMissing
                          ? 'border-red-500/40 bg-red-500/[0.02]'
                          : 'border-neuro-border'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-xs uppercase text-white">{mod.name}</span>
                          <span className="text-[10px] text-neuro-text-muted font-mono">
                            {mod.required ? '(Required)' : '(Optional Ground Truth)'}
                          </span>
                        </div>

                        {hasSelection ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                            <Check className="w-3 h-3" />
                            Detected
                          </span>
                        ) : isAmbiguous ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            {mod.candidates.length} Matches
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-red-500/10 text-red-400 border border-red-500/20 flex items-center gap-1">
                            <AlertCircle className="w-3 h-3" />
                            Missing
                          </span>
                        )}
                      </div>

                      <p className="text-[11px] text-neuro-text-secondary leading-tight">{mod.fullName}</p>

                      {/* File details or Ambiguity Resolver */}
                      {isAmbiguous ? (
                        <div className="space-y-1.5 pt-1">
                          <label className="text-[10px] text-amber-300 font-mono block">
                            Ambiguous match: Select correct file from folder:
                          </label>
                          <select
                            value={selectedFile?.name || ''}
                            onChange={e => handleSelectAmbiguousFile(key, e.target.value)}
                            className="w-full text-xs font-mono bg-neuro-surface-300 border border-amber-500/40 rounded px-2.5 py-1.5 text-white focus:outline-none focus:ring-1 focus:ring-amber-400"
                          >
                            <option value="">-- Choose file for {mod.name} --</option>
                            {mod.candidates.map(c => (
                              <option key={c.name} value={c.name}>
                                {c.name} ({formatFileSize(c.size)})
                              </option>
                            ))}
                          </select>
                        </div>
                      ) : selectedFile ? (
                        <div className="p-2 rounded bg-neuro-surface-300/80 border border-neuro-border flex items-center justify-between text-xs font-mono">
                          <span className="truncate max-w-[240px] text-white" title={selectedFile.name}>
                            {selectedFile.name}
                          </span>
                          <span className="text-[11px] text-neuro-text-muted flex-shrink-0 ml-2">
                            {formatFileSize(selectedFile.size)}
                          </span>
                        </div>
                      ) : (
                        <div className="p-2 rounded bg-red-500/5 border border-red-500/20 text-[11px] font-mono text-red-400">
                          No matching {mod.name} NIfTI file found in study folder.
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Spatial Alignment and NIfTI Integrity Verification Summary */}
              {validationReport && (
                <div
                  className={`p-4 rounded-xl border space-y-3 transition-all ${
                    validationReport.is_valid
                      ? 'bg-emerald-500/[0.03] border-emerald-500/30'
                      : 'bg-red-500/[0.03] border-red-500/30'
                  }`}
                >
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div className="flex items-center gap-2">
                      {validationReport.is_valid ? (
                        <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-red-400" />
                      )}
                      <div>
                        <h5 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
                          {validationReport.is_valid
                            ? 'Spatial Alignment & Study Verification: Passed'
                            : 'Study Verification: Action Required'}
                        </h5>
                        <p className="text-[11px] text-neuro-text-secondary mt-0.5">{validationReport.summary}</p>
                      </div>
                    </div>

                    <button
                      type="button"
                      disabled={isValidatingStudy}
                      onClick={() => runStudyValidation(selectedModalityFiles)}
                      className="px-2.5 py-1 rounded bg-neuro-surface-300 hover:bg-neuro-surface-400 border border-neuro-border text-[11px] font-mono text-neuro-text-secondary hover:text-white transition-colors flex items-center gap-1.5"
                    >
                      <RefreshCw className={`w-3 h-3 ${isValidatingStudy ? 'animate-spin' : ''}`} />
                      <span>Re-validate</span>
                    </button>
                  </div>

                  {/* Telemetry Chips */}
                  {validationReport.volume_dimensions && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-neuro-border/60 text-xs font-mono">
                      <div className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border">
                        <span className="text-[10px] text-neuro-text-muted uppercase block">Matrix Dimensions</span>
                        <span className="font-bold text-white">
                          {validationReport.volume_dimensions.join(' × ')}
                        </span>
                      </div>
                      <div className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border">
                        <span className="text-[10px] text-neuro-text-muted uppercase block">Voxel Spacing</span>
                        <span className="font-bold text-white">
                          {validationReport.voxel_spacing ? validationReport.voxel_spacing.map(z => z.toFixed(2)).join(' × ') : '1.00 × 1.00 × 1.00'} mm³
                        </span>
                      </div>
                      <div className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border">
                        <span className="text-[10px] text-neuro-text-muted uppercase block">Orientation</span>
                        <span className="font-bold text-cyan-400">
                          {validationReport.anatomical_orientation?.join('-') || 'RAS'}
                        </span>
                      </div>
                      <div className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border">
                        <span className="text-[10px] text-neuro-text-muted uppercase block">Spatial Alignment</span>
                        <span className={`font-bold ${validationReport.is_valid ? 'text-emerald-400' : 'text-red-400'}`}>
                          {validationReport.spatial_alignment?.is_aligned ? 'Affine Aligned' : 'Misaligned'}
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Errors / Warnings List */}
                  {validationReport.errors.length > 0 && (
                    <div className="space-y-1 pt-1 text-xs text-red-400">
                      {validationReport.errors.map((err, idx) => (
                        <p key={idx} className="flex items-center gap-1.5 font-mono text-[11px]">
                          <span>•</span>
                          <span>{err}</span>
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {error3D && (
            <div className="p-3.5 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-2.5 text-xs text-red-400">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error3D}</span>
            </div>
          )}

          {/* Action Button: Run 3D Segmentation */}
          <button
            type="button"
            disabled={!canRun3D}
            onClick={handleRun3DSegmentation}
            className={`w-full py-3 px-4 rounded-xl font-mono text-xs font-semibold uppercase tracking-wider flex items-center justify-center gap-2 transition-all shadow-md ${
              canRun3D
                ? 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold cursor-pointer shadow-cyan-500/20'
                : 'bg-neuro-surface-300 text-neuro-text-muted cursor-not-allowed opacity-60'
            }`}
          >
            {isAnalyzing3D ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-slate-950" />
                <span>{stageProgress3D || 'Executing 3D Volumetric Segmentation...'}</span>
              </>
            ) : (
              <>
                <Layers className="w-4 h-4" />
                <span>Run 3D Multimodal Segmentation</span>
              </>
            )}
          </button>

          <ResearchDisclaimer />
        </div>
      )}
    </motion.div>
  );
}
