import { useState } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import MRIImageViewer from '@/components/viewer/MRIImageViewer';
import { useAnalysis } from '@/context/AnalysisContext';
import { getDemoAnalyses } from '@/services/mockApi';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import {
  Sparkles,
  Info,
  CheckCircle2,
  AlertTriangle,
  Brain,
} from 'lucide-react';

export default function ExplainabilityPage() {
  const { currentAnalysis } = useAnalysis();
  const demoList = getDemoAnalyses();
  const [selectedAnalysisId, setSelectedAnalysisId] = useState(
    currentAnalysis?.id || demoList[0].id
  );
  const [selectedMethod, setSelectedMethod] = useState<'gradcam' | 'gradcam_pp' | 'segmentation'>('gradcam');

  const analysis =
    (currentAnalysis && currentAnalysis.id === selectedAnalysisId)
      ? currentAnalysis
      : demoList.find(d => d.id === selectedAnalysisId) || demoList[0];

  const getOverlayImage = () => {
    if (selectedMethod === 'gradcam') {
      return analysis.explanations.find(e => e.method === 'gradcam')?.overlayDataUrl || null;
    }
    if (selectedMethod === 'gradcam_pp') {
      return analysis.explanations.find(e => e.method === 'gradcam_pp')?.overlayDataUrl || null;
    }
    return analysis.segmentation?.overlayDataUrl || null;
  };

  const getOverlayLabel = () => {
    if (selectedMethod === 'gradcam') return 'Grad-CAM Heatmap';
    if (selectedMethod === 'gradcam_pp') return 'Grad-CAM++ Heatmap';
    return 'Segmentation Mask';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <PageHeader
        title="Explainability Studio"
        subtitle="Dedicated environment for visual inspectability, layer activation attribution, and spatial localization."
      >
        <div className="flex items-center gap-2">
          <label className="text-xs text-neuro-text-muted">Active Scan:</label>
          <select
            value={selectedAnalysisId}
            onChange={e => setSelectedAnalysisId(e.target.value)}
            className="neuro-input text-xs py-1 px-2.5 bg-neuro-surface-200"
          >
            {demoList.map(item => (
              <option key={item.id} value={item.id}>
                {item.fileName} ({item.predictedClass})
              </option>
            ))}
          </select>
        </div>
      </PageHeader>

      {/* Main Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Interactive Viewer */}
        <div className="lg:col-span-8 space-y-4">
          <MRIImageViewer
            originalImage={analysis.imageDataUrl}
            overlayImage={getOverlayImage()}
            overlayLabel={getOverlayLabel()}
            title={`Explainability Inspector — ${analysis.fileName}`}
            showComparisonOption={true}
          />

          {/* Quick Method Selector pills */}
          <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-neuro-surface border border-neuro-border rounded-lg">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-neuro-accent" />
              <span className="text-xs font-semibold text-neuro-text-primary">XAI Technique:</span>
            </div>

            <div className="flex items-center gap-2 text-xs">
              <button
                onClick={() => setSelectedMethod('gradcam')}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  selectedMethod === 'gradcam'
                    ? 'bg-neuro-accent text-neuro-bg font-semibold'
                    : 'bg-neuro-surface-200 text-neuro-text-secondary hover:text-neuro-text-primary'
                }`}
              >
                Grad-CAM
              </button>
              <button
                onClick={() => setSelectedMethod('gradcam_pp')}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  selectedMethod === 'gradcam_pp'
                    ? 'bg-neuro-accent text-neuro-bg font-semibold'
                    : 'bg-neuro-surface-200 text-neuro-text-secondary hover:text-neuro-text-primary'
                }`}
              >
                Grad-CAM++
              </button>
              <button
                onClick={() => setSelectedMethod('segmentation')}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  selectedMethod === 'segmentation'
                    ? 'bg-neuro-accent text-neuro-bg font-semibold'
                    : 'bg-neuro-surface-200 text-neuro-text-secondary hover:text-neuro-text-primary'
                }`}
              >
                Segmentation
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Educational & Methodological Explanations */}
        <div className="lg:col-span-4 space-y-4">
          {/* Target Metadata Card */}
          <div className="neuro-card space-y-3">
            <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
              <Brain className="w-4 h-4 text-neuro-accent" />
              <h3 className="text-sm font-semibold text-neuro-text-primary">Explanation Metadata</h3>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Target Class:</span>
                <span className="font-semibold text-neuro-accent-bright">{analysis.predictedClass}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Feature Layer:</span>
                <span className="font-mono text-neuro-text-secondary">layer4.2.conv3</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Status:</span>
                <span className="flex items-center gap-1 text-neuro-success">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Synthesized Successfully
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Resolution:</span>
                <span className="text-neuro-text-secondary">256 × 256 matrix</span>
              </div>
            </div>
          </div>

          {/* Deep-dive Explanation Panel */}
          <div className="neuro-card space-y-4">
            <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
              <Info className="w-4 h-4 text-neuro-cyan" />
              <h3 className="text-sm font-semibold text-neuro-text-primary">
                Method Interpretation Guide
              </h3>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <h4 className="font-semibold text-neuro-text-primary mb-1">What this method shows:</h4>
                <p className="text-neuro-text-secondary leading-relaxed">
                  Gradient attribution maps highlight spatial activation peaks that significantly drove the final classification logit for <strong>{analysis.predictedClass}</strong>.
                </p>
              </div>

              <div>
                <h4 className="font-semibold text-neuro-text-primary mb-1">What this method cannot establish:</h4>
                <p className="text-neuro-text-secondary leading-relaxed">
                  Saliency does not prove tissue malignancy, cell type, or true pathological margins. High saliency in background or ventricle boundaries may indicate confounding visual context.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 space-y-1">
                <div className="flex items-center gap-1.5 text-amber-400 font-semibold text-[11px]">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Clinical Constraint</span>
                </div>
                <p className="text-[11px] text-amber-300/80 leading-relaxed">
                  Grad-CAM maps are qualitative research representations and should not be used as surgical margins or radiation target volumes.
                </p>
              </div>
            </div>
          </div>

          <ResearchDisclaimer />
        </div>
      </div>
    </motion.div>
  );
}
