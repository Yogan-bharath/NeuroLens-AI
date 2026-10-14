import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  Image as ImageIcon,
  X,
  FileWarning,
  ArrowRight,
  CheckCircle2,
  Loader2,
  Brain,
} from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { useAnalysis } from '@/context/AnalysisContext';
import { useConfiguration } from '@/context/ConfigurationContext';
import { formatFileSize } from '@/lib/utils';
import { Sliders, Cpu } from 'lucide-react';

const ACCEPTED_TYPES = { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] };
const MAX_SIZE = 10 * 1024 * 1024; // 10MB

const stages = [
  { label: 'Uploading image', icon: Upload },
  { label: 'Preprocessing scan', icon: ImageIcon },
  { label: 'Running classification', icon: Brain },
  { label: 'Generating explanations', icon: CheckCircle2 },
  { label: 'Complete', icon: CheckCircle2 },
];

export default function NewAnalysisPage() {
  const navigate = useNavigate();
  const { runAnalysis, isAnalyzing, statusStage } = useAnalysis();
  const { config, currentClassificationModel, currentSegmentationModel, currentXaiMethod } = useConfiguration();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ w: number; h: number } | null>(null);

  const onDrop = useCallback((accepted: File[], rejected: FileRejection[]) => {
    setError(null);
    if (rejected.length > 0) {
      const msg = rejected[0].errors[0]?.message || 'Invalid file';
      if (msg.includes('large')) setError('File exceeds 10 MB limit.');
      else setError('Unsupported file type. Please use PNG or JPEG.');
      return;
    }
    if (accepted.length > 0) {
      const file = accepted[0];
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => {
        const url = reader.result as string;
        setPreview(url);
        const img = new Image();
        img.onload = () => setDimensions({ w: img.width, h: img.height });
        img.src = url;
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxSize: MAX_SIZE,
    multiple: false,
    disabled: isAnalyzing,
  });

  const handleRemove = () => {
    setSelectedFile(null);
    setPreview(null);
    setDimensions(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!preview || !selectedFile) return;
    try {
      const result = await runAnalysis(preview, selectedFile.name, config);
      navigate(`/analysis/${result.id}`);
    } catch {
      setError('Analysis failed. Please try again.');
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
      <PageHeader
        title="New Analysis"
        subtitle="Upload a brain MRI scan to begin AI-powered analysis and explainability."
      >
        <button
          onClick={() => navigate('/model-selection')}
          className="neuro-btn-secondary text-xs flex items-center gap-1.5"
        >
          <Sliders className="w-3.5 h-3.5 text-neuro-accent" />
          <span>Model Selection</span>
        </button>
      </PageHeader>

      <div className="max-w-2xl mx-auto space-y-4">
        {/* Active Model Configuration Pill Bar */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-3 rounded-lg bg-neuro-surface-200/80 border border-neuro-border text-xs">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-neuro-accent flex-shrink-0" />
            <div>
              <span className="font-semibold text-neuro-text-primary capitalize">
                {config.mode} Mode:
              </span>{' '}
              <span className="text-neuro-text-secondary">
                {config.mode !== 'segmentation' && (
                  <>
                    <strong className="text-neuro-accent-bright">{currentClassificationModel?.name || 'EfficientNetV2B0'}</strong>{' '}
                    ({currentXaiMethod?.name || 'Grad-CAM'})
                  </>
                )}
                {config.mode === 'combined' && ' + '}
                {config.mode !== 'classification' && (
                  <>
                    <strong className="text-neuro-cyan">{currentSegmentationModel?.name || 'U-Net'}</strong>
                  </>
                )}
              </span>
            </div>
          </div>

          <button
            onClick={() => navigate('/model-selection')}
            className="text-[11px] font-medium text-neuro-accent hover:text-neuro-accent-bright hover:underline"
          >
            Change Models
          </button>
        </div>
        {/* Upload area */}
        {!isAnalyzing && (
          <AnimatePresence mode="wait">
            {!selectedFile ? (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div
                  {...getRootProps()}
                  className={`neuro-panel p-8 sm:p-12 flex flex-col items-center justify-center cursor-pointer
                    transition-all duration-300 border-dashed border-2
                    ${isDragActive
                      ? 'border-neuro-accent bg-neuro-accent-muted shadow-neuro-glow'
                      : 'border-neuro-border hover:border-neuro-border-hover hover:shadow-neuro-sm'
                    }`}
                >
                  <input {...getInputProps()} />
                  <div className="w-16 h-16 rounded-2xl bg-neuro-surface-300 flex items-center justify-center mb-4">
                    <Upload className={`w-8 h-8 ${isDragActive ? 'text-neuro-accent' : 'text-neuro-text-muted'}`} />
                  </div>
                  <p className="text-base font-medium text-neuro-text-primary mb-1">
                    {isDragActive ? 'Drop your MRI scan here' : 'Drag & drop your MRI scan'}
                  </p>
                  <p className="text-sm text-neuro-text-muted mb-4">
                    or click to browse files
                  </p>
                  <button className="neuro-btn-secondary text-sm" type="button">
                    <ImageIcon className="w-4 h-4" />
                    Browse Files
                  </button>
                  <p className="text-xs text-neuro-text-disabled mt-4">
                    Supported: PNG, JPEG · Max size: 10 MB · 2D brain MRI slices
                  </p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="preview"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="neuro-panel overflow-hidden"
              >
                {/* Image preview */}
                <div className="relative bg-neuro-bg p-4 flex items-center justify-center">
                  <img
                    src={preview!}
                    alt="MRI Preview"
                    className="max-h-72 rounded-md object-contain"
                  />
                  <button
                    onClick={handleRemove}
                    className="absolute top-3 right-3 p-1.5 rounded-md bg-neuro-surface-300/80 hover:bg-neuro-error/20 text-neuro-text-muted hover:text-neuro-error transition-colors"
                    aria-label="Remove file"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* File info */}
                <div className="p-4 border-t border-neuro-border">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-lg bg-neuro-surface-200 flex items-center justify-center">
                      <ImageIcon className="w-5 h-5 text-neuro-accent" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-neuro-text-primary truncate">{selectedFile.name}</p>
                      <p className="text-xs text-neuro-text-muted">
                        {formatFileSize(selectedFile.size)}
                        {dimensions && ` · ${dimensions.w} × ${dimensions.h} px`}
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <button onClick={handleRemove} className="neuro-btn-secondary flex-1">
                      Replace
                    </button>
                    <button onClick={handleAnalyze} className="neuro-btn-primary flex-1">
                      <Brain className="w-4 h-4" />
                      Analyze MRI
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        )}

        {/* Processing state */}
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="neuro-panel p-8"
          >
            <div className="flex flex-col items-center mb-6">
              <div className="relative w-16 h-16 mb-4">
                <div className="absolute inset-0 rounded-full border-2 border-neuro-surface-300" />
                <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-neuro-accent animate-spin" />
                <Brain className="absolute inset-0 m-auto w-6 h-6 text-neuro-accent" />
              </div>
              <p className="text-sm font-medium text-neuro-text-primary">Analyzing your MRI scan...</p>
              <p className="text-xs text-neuro-text-muted mt-1">This is a demo simulation</p>
            </div>

            <div className="space-y-2">
              {stages.map((stage, i) => {
                const isComplete = i < statusStage;
                const isCurrent = i === statusStage;
                return (
                  <div key={i} className="flex items-center gap-3">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 ${
                      isComplete ? 'bg-neuro-success/20' :
                      isCurrent ? 'bg-neuro-accent-muted' :
                      'bg-neuro-surface-200'
                    }`}>
                      {isComplete ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-neuro-success" />
                      ) : isCurrent ? (
                        <Loader2 className="w-3.5 h-3.5 text-neuro-accent animate-spin" />
                      ) : (
                        <div className="w-1.5 h-1.5 rounded-full bg-neuro-text-disabled" />
                      )}
                    </div>
                    <span className={`text-sm ${
                      isComplete ? 'text-neuro-text-secondary' :
                      isCurrent ? 'text-neuro-text-primary font-medium' :
                      'text-neuro-text-disabled'
                    }`}>
                      {stage.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}

        {/* Error */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 flex items-center gap-2 p-3 rounded-lg bg-neuro-error/10 border border-neuro-error/20"
          >
            <FileWarning className="w-4 h-4 text-neuro-error flex-shrink-0" />
            <p className="text-sm text-red-300">{error}</p>
          </motion.div>
        )}

        {/* Info */}
        <div className="mt-6">
          <ResearchDisclaimer />
        </div>
        <p className="text-xs text-neuro-text-disabled mt-3 text-center">
          In demo mode, all analyses use predefined mock results. No real inference is performed.
        </p>
      </div>
    </motion.div>
  );
}
