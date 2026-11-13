import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAnalysis } from '@/context/AnalysisContext';
import MRIImageViewer from '@/components/viewer/MRIImageViewer';
import ClassificationResults from '@/components/analysis/ClassificationResults';
import ExplainabilityTabs, { type ActiveTabType } from '@/components/analysis/ExplainabilityTabs';
import ResultsSummary from '@/components/analysis/ResultsSummary';
import AIChatPanel from '@/components/chat/AIChatPanel';
import PageHeader from '@/components/layout/PageHeader';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ArrowLeft, MessageSquare, Sparkles, AlertCircle } from 'lucide-react';
import { getDemoAnalyses } from '@/services/mockApi';
import { getAnalysisById } from '@/services/api';

export default function AnalysisWorkspacePage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { currentAnalysis, loadAnalysis } = useAnalysis();
  const [activeTab, setActiveTab] = useState<ActiveTabType>('gradcam');
  const [showChatDrawer, setShowChatDrawer] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (id) {
      setIsLoading(true);
      setError(null);
      getAnalysisById(id)
        .then(analysis => {
          if (analysis) {
            const is3D =
              analysis.workflow_type === '3D_SEGMENTATION' ||
              analysis.analysis_mode === 'segmentation_3d' ||
              analysis.analysis_mode === 'segmentation' ||
              Boolean(analysis.surface_3d) ||
              Boolean(analysis.volume_dimensions);

            if (is3D) {
              navigate(`/analysis/${id}/3d-results`, { replace: true });
            } else {
              navigate(`/analysis/${id}/2d-results`, { replace: true });
            }
          } else {
            setError(`Analysis session '${id}' was not found.`);
          }
        })
        .catch(err => {
          setError(err instanceof Error ? err.message : `Failed to retrieve analysis '${id}'.`);
        })
        .finally(() => setIsLoading(false));
    } else {
      navigate('/history', { replace: true });
    }
  }, [id, navigate]);

  if (isLoading) {
    return <LoadingSpinner text="Directing to Verified Analysis Results..." className="min-h-[500px]" />;
  }

  if (error || !currentAnalysis) {
    return (
      <div className="max-w-xl mx-auto py-16 px-4">
        <div className="p-8 rounded-xl bg-neuro-surface-200 border border-neuro-border text-center space-y-4 shadow-sm">
          <AlertCircle className="w-10 h-10 text-red-400 mx-auto" />
          <h3 className="text-lg font-bold text-white">Analysis Not Found</h3>
          <p className="text-xs text-neuro-text-secondary leading-relaxed font-mono">
            {error || `The requested analysis session '${id}' could not be located in storage.`}
          </p>
          <div className="pt-2 flex justify-center gap-3">
            <button
              onClick={() => navigate('/history')}
              className="px-4 py-2 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-secondary text-xs font-semibold border border-neuro-border"
            >
              View Analysis History
            </button>
            <button
              onClick={() => navigate('/new-analysis')}
              className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-semibold"
            >
              Start New Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }

  const analysis = currentAnalysis!;

  // Active overlay selection based on tab
  const getActiveOverlayImage = () => {
    if (activeTab === 'gradcam') {
      return analysis.explanations.find(e => e.method === 'gradcam')?.overlayDataUrl || null;
    }
    if (activeTab === 'gradcam_pp') {
      return analysis.explanations.find(e => e.method === 'gradcam_pp')?.overlayDataUrl || null;
    }
    if (activeTab === 'segmentation') {
      return analysis.segmentation?.overlayDataUrl || null;
    }
    // Comparison mode defaults to Grad-CAM overlay
    return analysis.explanations[0]?.overlayDataUrl || null;
  };

  const getActiveOverlayLabel = () => {
    switch (activeTab) {
      case 'gradcam':
        return 'Grad-CAM Saliency';
      case 'gradcam_pp':
        return 'Grad-CAM++ Saliency';
      case 'segmentation':
        return 'Segmentation Mask';
      case 'comparison':
        return 'Side-by-Side Comparison';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Page Header */}
      <PageHeader
        title="MRI Analysis Workspace"
        subtitle={`Session ID: ${analysis.id} · File: ${analysis.fileName}`}
      >
        <button
          onClick={() => navigate('/history')}
          className="neuro-btn-secondary text-xs"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>All Analyses</span>
        </button>

        <button
          onClick={() => setShowChatDrawer(prev => !prev)}
          className={`neuro-btn-secondary text-xs ${
            showChatDrawer ? 'bg-neuro-accent-muted text-neuro-accent-bright border-neuro-accent/40' : ''
          }`}
        >
          <MessageSquare className="w-4 h-4" />
          <span>{showChatDrawer ? 'Hide Copilot' : 'AI Copilot'}</span>
        </button>
      </PageHeader>

      {/* Top Results Summary Bar */}
      <ResultsSummary
        analysis={analysis}
        selectedMethod={getActiveOverlayLabel()}
      />

      {/* Main 3-Part Responsive Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Center / Left Column: Large MRI Viewer (7 cols or 8 cols depending on chat) */}
        <div className={`space-y-4 ${showChatDrawer ? 'lg:col-span-7' : 'lg:col-span-8'}`}>
          <MRIImageViewer
            originalImage={analysis.imageDataUrl}
            overlayImage={getActiveOverlayImage()}
            overlayLabel={getActiveOverlayLabel()}
            title={analysis.fileName}
            showComparisonOption={true}
          />

          {/* Explainability Sub-Panel under viewer */}
          <ExplainabilityTabs
            activeTab={activeTab}
            onChangeTab={setActiveTab}
            explanations={analysis.explanations}
            segmentation={analysis.segmentation}
            targetClass={analysis.predictedClass || 'Unknown'}
          />
        </div>

        {/* Right Column: Classification Results + Optional Chat */}
        <div className={`space-y-4 ${showChatDrawer ? 'lg:col-span-5' : 'lg:col-span-4'}`}>
          {showChatDrawer ? (
            /* When chat drawer is active in right column */
            <div className="space-y-4">
              <AIChatPanel
                analysisId={analysis.id}
                predictedClass={analysis.predictedClass}
                hasSegmentation={Boolean(analysis.segmentation)}
              />
              <ClassificationResults
                predictedClass={analysis.predictedClass || 'Glioma'}
                confidence={analysis.confidence || 0.85}
                scores={analysis.classification || []}
                metadata={analysis.metadata}
              />
            </div>
          ) : (
            /* Default column view with classification details */
            <div className="space-y-4">
              <ClassificationResults
                predictedClass={analysis.predictedClass || 'Glioma'}
                confidence={analysis.confidence || 0.85}
                scores={analysis.classification || []}
                metadata={analysis.metadata}
              />

              <div className="neuro-card p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-neuro-accent" />
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-neuro-text-primary">
                    AI Research Copilot
                  </h4>
                </div>
                <p className="text-xs text-neuro-text-secondary leading-relaxed">
                  Have questions about the Grad-CAM activation, model confidence, or feature patterns for {analysis.predictedClass}? Open the integrated AI Copilot.
                </p>
                <button
                  onClick={() => setShowChatDrawer(true)}
                  className="neuro-btn-primary w-full text-xs py-2"
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  Launch Copilot Chat
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
