import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Upload,
  ArrowRight,
  Brain,
  Sparkles,
  BarChart3,
  Eye,
  Activity,
  FileSearch,
  Layers,
  Clock,
  Sliders,
} from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import MetricCard from '@/components/common/MetricCard';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { getDemoAnalyses } from '@/services/mockApi';
import { formatDate, formatPercentage } from '@/lib/utils';

const stagger = {
  container: { transition: { staggerChildren: 0.06 } },
  item: {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35 } },
  },
};

const workflowSteps = [
  { icon: Upload, title: 'Upload MRI', desc: 'Upload a brain MRI scan in PNG or JPEG format' },
  { icon: Brain, title: 'AI Classification', desc: 'Model predicts tumor type with confidence scores' },
  { icon: Sparkles, title: 'Explainability', desc: 'Grad-CAM heatmaps show decision-driving regions' },
  { icon: Eye, title: 'Review & Export', desc: 'Inspect results, chat with AI, export reports' },
];

export default function OverviewPage() {
  const navigate = useNavigate();
  const demoAnalyses = getDemoAnalyses();

  return (
    <motion.div variants={stagger.container} initial="initial" animate="animate">
      {/* Hero */}
      <motion.div variants={stagger.item} className="mb-8">
        <PageHeader
          title="Brain MRI Analysis"
          subtitle="Explore model predictions, visual explanations and segmentation results in one workspace."
        >
          <button onClick={() => navigate('/new-analysis')} className="neuro-btn-primary">
            <Upload className="w-4 h-4" />
            Start New Analysis
          </button>
          <button onClick={() => navigate('/model-selection')} className="neuro-btn-secondary">
            <Sliders className="w-4 h-4 text-neuro-accent" />
            Model Selection
          </button>
          <button
            onClick={() => {
              const demoId = demoAnalyses[0]?.id;
              if (demoId) navigate(`/analysis/${demoId}`);
            }}
            className="neuro-btn-secondary"
          >
            Explore Demo
            <ArrowRight className="w-4 h-4" />
          </button>
        </PageHeader>
      </motion.div>

      {/* Metrics */}
      <motion.div variants={stagger.item} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Total Analyses"
          value={demoAnalyses.length}
          icon={FileSearch}
          trend="Demo records"
          accent="teal"
        />
        <MetricCard
          label="Model Accuracy"
          value="92.3%"
          icon={Activity}
          trend="Example metric"
          accent="cyan"
        />
        <MetricCard
          label="Supported Classes"
          value={4}
          icon={Layers}
          trend="Glioma, Meningioma, Pituitary, No Tumor"
          accent="blue"
        />
        <MetricCard
          label="Avg. Processing"
          value="~3.6s"
          icon={Clock}
          trend="Demo estimate"
          accent="amber"
        />
      </motion.div>

      {/* Upload quick-action */}
      <motion.div variants={stagger.item} className="mb-8">
        <button
          onClick={() => navigate('/new-analysis')}
          className="w-full neuro-panel p-6 flex items-center gap-5 hover:border-neuro-accent/30 hover:shadow-neuro-glow transition-all duration-300 group cursor-pointer"
        >
          <div className="w-12 h-12 rounded-xl bg-neuro-accent-muted flex items-center justify-center group-hover:scale-105 transition-transform">
            <Upload className="w-6 h-6 text-neuro-accent" />
          </div>
          <div className="text-left">
            <p className="text-sm font-medium text-neuro-text-primary">Upload a Brain MRI Scan</p>
            <p className="text-xs text-neuro-text-muted mt-0.5">PNG or JPEG · Max 10 MB · 2D slices supported</p>
          </div>
          <ArrowRight className="w-5 h-5 text-neuro-text-muted ml-auto group-hover:translate-x-1 transition-transform" />
        </button>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Recent analyses */}
        <motion.div variants={stagger.item} className="lg:col-span-2">
          <h3 className="text-sm font-medium text-neuro-text-secondary mb-3">Recent Analyses</h3>
          <div className="space-y-2">
            {demoAnalyses.map(analysis => (
              <button
                key={analysis.id}
                onClick={() => navigate(`/analysis/${analysis.id}`)}
                className="w-full neuro-card-hover flex items-center gap-4 text-left cursor-pointer"
              >
                <div className="w-12 h-12 rounded-lg bg-neuro-surface-200 overflow-hidden flex-shrink-0">
                  <img
                    src={analysis.imageDataUrl}
                    alt={analysis.fileName}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-neuro-text-primary truncate">{analysis.fileName}</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="neuro-badge-accent">{analysis.predictedClass}</span>
                    <span className="text-xs text-neuro-text-muted">
                      {analysis.confidence ? formatPercentage(analysis.confidence) : '—'}
                    </span>
                  </div>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className="text-xs text-neuro-text-muted">{formatDate(analysis.createdAt)}</p>
                  <span className="neuro-badge-success mt-1">Demo</span>
                </div>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Model info */}
        <motion.div variants={stagger.item}>
          <h3 className="text-sm font-medium text-neuro-text-secondary mb-3">Model Overview</h3>
          <div className="neuro-card space-y-3">
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-neuro-accent" />
              <span className="text-sm font-medium text-neuro-text-primary">NeuroClassifier v2.1.0</span>
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Architecture</span>
                <span className="text-neuro-text-secondary">ResNet-50 (Modified)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Framework</span>
                <span className="text-neuro-text-secondary">PyTorch 2.1</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Input Size</span>
                <span className="text-neuro-text-secondary">256 × 256 × 1</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Parameters</span>
                <span className="text-neuro-text-secondary">23.5M</span>
              </div>
            </div>
            <p className="text-xs text-neuro-text-disabled pt-1 border-t border-neuro-border">
              Example model metadata — not from a real evaluation
            </p>
          </div>
        </motion.div>
      </div>

      {/* Workflow */}
      <motion.div variants={stagger.item} className="mb-8">
        <h3 className="text-sm font-medium text-neuro-text-secondary mb-4">How It Works</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {workflowSteps.map((step, i) => (
            <div key={i} className="neuro-card relative">
              <div className="absolute top-3 right-3 w-6 h-6 rounded-full bg-neuro-surface-200 flex items-center justify-center">
                <span className="text-xs font-bold text-neuro-text-muted">{i + 1}</span>
              </div>
              <step.icon className="w-8 h-8 text-neuro-accent mb-3" />
              <h4 className="text-sm font-medium text-neuro-text-primary">{step.title}</h4>
              <p className="text-xs text-neuro-text-muted mt-1">{step.desc}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div variants={stagger.item}>
        <ResearchDisclaimer />
      </motion.div>
    </motion.div>
  );
}
