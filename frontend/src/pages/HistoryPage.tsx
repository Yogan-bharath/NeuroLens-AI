import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import AnalysisHistoryTable from '@/components/analysis/AnalysisHistoryTable';
import { getAnalysisHistory } from '@/services/api';
import type { AnalysisSummary } from '@/types/analysis';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { Info, Plus, ShieldCheck, Lock, LogIn, RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';

export default function HistoryPage() {
  const navigate = useNavigate();
  const { user, isAuthenticated, openAuthModal } = useAuth();
  const [records, setRecords] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getAnalysisHistory();
      setRecords(data);
    } catch {
      setRecords([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory, isAuthenticated]);

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
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchHistory()}
            className="neuro-btn-secondary text-xs px-2.5 py-1.5 flex items-center gap-1.5"
            title="Refresh history"
          >
            <RefreshCw className="w-3.5 h-3.5 text-neuro-text-muted" />
            <span>Refresh</span>
          </button>
          <button
            onClick={() => navigate('/new-analysis')}
            className="neuro-btn-primary text-xs"
          >
            <Plus className="w-4 h-4" />
            <span>New Analysis</span>
          </button>
        </div>
      </PageHeader>

      {/* Account isolation & security status */}
      {isAuthenticated && user ? (
        <div className="flex items-center justify-between p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs">
          <div className="flex items-center gap-2 text-emerald-400">
            <ShieldCheck className="w-4 h-4 flex-shrink-0" />
            <span>
              Secure Private Workspace: Viewing analyses strictly bound to{' '}
              <strong className="text-white">{user.name}</strong> ({user.email}).
            </span>
          </div>
          <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold">
            Isolated Account
          </span>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border text-xs">
          <div className="flex items-start gap-2.5">
            <Lock className="w-4 h-4 text-neuro-accent flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-neuro-text-primary">
                Account Sign In Required for Saved Reports
              </p>
              <p className="text-neuro-text-secondary text-[11px] mt-0.5">
                Clinical MRI analyses are private and bound exclusively to individual researcher accounts to prevent unauthorized access.
              </p>
            </div>
          </div>
          <button
            onClick={() => openAuthModal('login')}
            className="neuro-btn-primary text-xs flex items-center gap-1.5 flex-shrink-0"
          >
            <LogIn className="w-3.5 h-3.5" />
            <span>Sign In to View Reports</span>
          </button>
        </div>
      )}

      {loading ? (
        <LoadingSpinner text="Retrieving analysis history..." className="py-16" />
      ) : (
        <AnalysisHistoryTable
          initialRecords={records}
          onDeleteSuccess={() => fetchHistory()}
        />
      )}

      <ResearchDisclaimer />
    </motion.div>
  );
}
