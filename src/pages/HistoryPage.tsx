import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import AnalysisHistoryTable from '@/components/analysis/AnalysisHistoryTable';
import { mockGetAnalysisHistory } from '@/services/mockApi';
import type { AnalysisSummary } from '@/types/analysis';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { Info, Plus } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function HistoryPage() {
  const navigate = useNavigate();
  const [records, setRecords] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    mockGetAnalysisHistory()
      .then(data => setRecords(data))
      .finally(() => setLoading(false));
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <PageHeader
        title="Analysis History"
        subtitle="Catalog of recent brain MRI inference sessions and explainability records."
      >
        <button
          onClick={() => navigate('/new-analysis')}
          className="neuro-btn-primary text-xs"
        >
          <Plus className="w-4 h-4" />
          <span>New Analysis</span>
        </button>
      </PageHeader>

      {/* Demo Notice */}
      <div className="flex items-start gap-2.5 p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs text-neuro-text-secondary">
        <Info className="w-4 h-4 text-neuro-accent flex-shrink-0 mt-0.5" />
        <p>
          <strong className="text-neuro-text-primary">Session Persistence Notice:</strong>{' '}
          In demo mode, analysis history records exist purely in browser memory during this demo session and are not stored in any external database.
        </p>
      </div>

      {loading ? (
        <LoadingSpinner text="Retrieving analysis history..." className="py-16" />
      ) : (
        <AnalysisHistoryTable initialRecords={records} />
      )}

      <ResearchDisclaimer />
    </motion.div>
  );
}
