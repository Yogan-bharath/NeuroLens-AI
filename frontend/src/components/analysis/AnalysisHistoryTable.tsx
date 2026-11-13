import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AnalysisSummary } from '@/types/analysis';
import { formatDate, formatPercentage } from '@/lib/utils';
import { deleteAnalysis } from '@/services/api';
import {
  ExternalLink,
  Trash2,
  Search,
  Filter,
  ArrowUpDown,
  AlertTriangle,
  Loader2,
} from 'lucide-react';

interface AnalysisHistoryTableProps {
  initialRecords: AnalysisSummary[];
  onDeleteSuccess?: (id: string) => void;
}

export default function AnalysisHistoryTable({ initialRecords, onDeleteSuccess }: AnalysisHistoryTableProps) {
  const navigate = useNavigate();
  const [records, setRecords] = useState<AnalysisSummary[]>(initialRecords);
  const [searchTerm, setSearchTerm] = useState('');
  const [classFilter, setClassFilter] = useState('all');
  const [deleteCandidateId, setDeleteCandidateId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    setRecords(initialRecords);
  }, [initialRecords]);

  // Filtered and searched records
  const filtered = records
    .filter(item => {
      const matchesSearch =
        item.fileName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (item.predictedClass || '').toLowerCase().includes(searchTerm.toLowerCase());

      const matchesClass =
        classFilter === 'all' || item.predictedClass?.toLowerCase() === classFilter.toLowerCase();

      return matchesSearch && matchesClass;
    })
    .sort((a, b) => {
      const dateA = new Date(a.createdAt).getTime();
      const dateB = new Date(b.createdAt).getTime();
      return sortAsc ? dateA - dateB : dateB - dateA;
    });

  const handleDelete = async (id: string) => {
    try {
      setIsDeleting(true);
      setDeleteError(null);
      await deleteAnalysis(id);
      setRecords(prev => prev.filter(r => r.id !== id));
      setDeleteCandidateId(null);
      if (onDeleteSuccess) {
        onDeleteSuccess(id);
      }
    } catch (err: any) {
      setDeleteError(err.message || 'Failed to delete record.');
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Search and Filters toolbar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-neuro-text-muted absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            placeholder="Search by ID, file name, or predicted class..."
            className="neuro-input pl-9 w-full text-xs"
          />
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 bg-neuro-surface-200 border border-neuro-border rounded-md px-2.5 py-1.5 text-xs text-neuro-text-secondary">
            <Filter className="w-3.5 h-3.5 text-neuro-text-muted" />
            <select
              value={classFilter}
              onChange={e => setClassFilter(e.target.value)}
              className="bg-transparent text-xs text-neuro-text-primary focus:outline-none cursor-pointer"
            >
              <option value="all">All Classes</option>
              <option value="glioma">Glioma</option>
              <option value="meningioma">Meningioma</option>
              <option value="pituitary">Pituitary</option>
              <option value="no tumor">No Tumor</option>
            </select>
          </div>

          <button
            onClick={() => setSortAsc(prev => !prev)}
            className="neuro-btn-secondary text-xs px-2.5 py-1.5"
            title="Sort by date"
          >
            <ArrowUpDown className="w-3.5 h-3.5" />
            <span>{sortAsc ? 'Oldest' : 'Newest'}</span>
          </button>
        </div>
      </div>

      {/* Table view */}
      <div className="neuro-card overflow-hidden p-0 border border-neuro-border">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-neuro-border bg-neuro-surface-200 text-neuro-text-muted uppercase text-[10px] tracking-wider font-semibold">
                <th className="py-3 px-4">Preview</th>
                <th className="py-3 px-4">Workflow</th>
                <th className="py-3 px-4">Analysis ID</th>
                <th className="py-3 px-4">File Name</th>
                <th className="py-3 px-4">Predicted Class / Result</th>
                <th className="py-3 px-4">Confidence / Volume</th>
                <th className="py-3 px-4">Date</th>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neuro-border">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-8 text-center text-neuro-text-muted">
                    No matching analysis records found.
                  </td>
                </tr>
              ) : (
                filtered.map(record => {
                  const isCombined =
                    record.workflow_type === 'COMBINED_ANALYSIS' ||
                    record.analysis_mode === 'combined' ||
                    record.id.startsWith('CMB-');
                  const is3D =
                    record.workflow_type === '3D_SEGMENTATION' ||
                    record.analysis_mode === 'segmentation_3d' ||
                    record.analysis_mode === 'segmentation' ||
                    Boolean((record as any).volume_dimensions) ||
                    Boolean((record as any).surface_3d) ||
                    (record.predictedClass || '').toLowerCase().includes('segmentation') ||
                    (record.predictedClass || '').toLowerCase().includes('multimodal');
                  const targetRoute = isCombined
                    ? `/analysis/${record.id}/combined-results`
                    : is3D
                    ? `/analysis/${record.id}/3d-results`
                    : `/analysis/${record.id}/2d-results`;

                  return (
                    <tr
                      key={record.id}
                      onClick={() => navigate(targetRoute)}
                      className="hover:bg-neuro-surface-200/50 transition-colors cursor-pointer"
                    >
                      <td className="py-2.5 px-4">
                        <div className="w-10 h-10 rounded bg-neuro-surface-300 overflow-hidden flex items-center justify-center border border-neuro-border/60">
                          <img
                            src={record.thumbnailUrl}
                            alt={record.fileName}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </td>
                      <td className="py-2.5 px-4">
                        {isCombined ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20 whitespace-nowrap">
                            Dual Combined
                          </span>
                        ) : is3D ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 whitespace-nowrap">
                            3D Segmentation
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 whitespace-nowrap">
                            2D Classification
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-4 font-mono font-medium text-neuro-accent-bright">
                        {record.id}
                      </td>
                      <td className="py-2.5 px-4 font-medium text-neuro-text-primary">
                        {record.fileName}
                      </td>
                      <td className="py-2.5 px-4">
                        <span className={is3D ? 'px-2 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-300 border border-cyan-500/20' : 'neuro-badge-accent'}>
                          {record.predictedClass || (is3D ? '3D Tumor Mask' : 'Unknown')}
                        </span>
                      </td>
                      <td className="py-2.5 px-4 font-mono font-semibold text-neuro-text-primary">
                        {record.confidence ? formatPercentage(record.confidence) : (is3D ? 'Volumetric' : '—')}
                      </td>
                      <td className="py-2.5 px-4 text-neuro-text-secondary whitespace-nowrap">
                        {formatDate(record.createdAt)}
                      </td>
                      <td className="py-2.5 px-4 text-neuro-text-muted font-mono">
                        <div className="flex flex-col">
                          <span className="font-medium text-neuro-text-primary">
                            {record.actual_model?.model_name || record.model_name || record.modelVersion}
                          </span>
                          {(record.actual_model?.model_id || record.model_id) && (
                            <span className="text-[10px] text-neuro-accent">
                              {record.actual_model?.model_id || record.model_id}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-2.5 px-4 text-right" onClick={e => e.stopPropagation()}>
                        <div className="inline-flex items-center gap-1.5">
                          <button
                            onClick={() => navigate(targetRoute)}
                            className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-accent hover:text-neuro-accent-bright transition-colors"
                            title="Open Dedicated Results Page"
                          >
                            <ExternalLink className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setDeleteCandidateId(record.id)}
                            className="p-1.5 rounded hover:bg-neuro-error/15 text-neuro-text-muted hover:text-neuro-error transition-colors"
                            title="Delete Record"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteCandidateId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
          <div className="w-full max-w-sm bg-neuro-surface border border-neuro-border rounded-xl p-6 space-y-4 shadow-2xl">
            <div className="flex items-center gap-2 text-neuro-error font-semibold text-sm">
              <AlertTriangle className="w-5 h-5" />
              <span>Confirm Record Deletion</span>
            </div>
            <p className="text-xs text-neuro-text-secondary leading-relaxed">
              Are you sure you want to permanently delete analysis session{' '}
              <strong className="text-neuro-text-primary font-mono">{deleteCandidateId}</strong> and all associated cloud storage artifacts?
            </p>
            {deleteError && (
              <div className="p-2.5 rounded bg-neuro-error/15 border border-neuro-error/30 text-neuro-error text-xs">
                {deleteError}
              </div>
            )}
            <div className="flex justify-end gap-2 pt-2">
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => {
                  setDeleteCandidateId(null);
                  setDeleteError(null);
                }}
                className="neuro-btn-secondary text-xs"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isDeleting}
                onClick={() => handleDelete(deleteCandidateId)}
                className="neuro-btn-primary bg-neuro-error hover:bg-red-600 text-white text-xs flex items-center gap-1.5 disabled:opacity-50"
              >
                {isDeleting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                <span>{isDeleting ? 'Deleting...' : 'Delete Permanently'}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
