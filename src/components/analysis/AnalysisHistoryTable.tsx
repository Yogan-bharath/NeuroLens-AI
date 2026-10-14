import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AnalysisSummary } from '@/types/analysis';
import { formatDate, formatPercentage } from '@/lib/utils';
import {
  ExternalLink,
  Trash2,
  Search,
  Filter,
  ArrowUpDown,
  AlertTriangle,
} from 'lucide-react';

interface AnalysisHistoryTableProps {
  initialRecords: AnalysisSummary[];
}

export default function AnalysisHistoryTable({ initialRecords }: AnalysisHistoryTableProps) {
  const navigate = useNavigate();
  const [records, setRecords] = useState<AnalysisSummary[]>(initialRecords);
  const [searchTerm, setSearchTerm] = useState('');
  const [classFilter, setClassFilter] = useState('all');
  const [deleteCandidateId, setDeleteCandidateId] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(false);

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

  const handleDelete = (id: string) => {
    setRecords(prev => prev.filter(r => r.id !== id));
    setDeleteCandidateId(null);
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
                <th className="py-3 px-4">Analysis ID</th>
                <th className="py-3 px-4">File Name</th>
                <th className="py-3 px-4">Predicted Class</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Date</th>
                <th className="py-3 px-4">Model</th>
                <th className="py-3 px-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neuro-border">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-8 text-center text-neuro-text-muted">
                    No matching analysis records found.
                  </td>
                </tr>
              ) : (
                filtered.map(record => (
                  <tr
                    key={record.id}
                    className="hover:bg-neuro-surface-200/50 transition-colors"
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
                    <td className="py-2.5 px-4 font-mono font-medium text-neuro-accent-bright">
                      {record.id}
                    </td>
                    <td className="py-2.5 px-4 font-medium text-neuro-text-primary">
                      {record.fileName}
                    </td>
                    <td className="py-2.5 px-4">
                      <span className="neuro-badge-accent">
                        {record.predictedClass || 'Unknown'}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 font-mono font-semibold text-neuro-text-primary">
                      {record.confidence ? formatPercentage(record.confidence) : '—'}
                    </td>
                    <td className="py-2.5 px-4 text-neuro-text-secondary whitespace-nowrap">
                      {formatDate(record.createdAt)}
                    </td>
                    <td className="py-2.5 px-4 text-neuro-text-muted font-mono">
                      {record.modelVersion}
                    </td>
                    <td className="py-2.5 px-4 text-right">
                      <div className="inline-flex items-center gap-1.5">
                        <button
                          onClick={() => navigate(`/analysis/${record.id}`)}
                          className="p-1.5 rounded hover:bg-neuro-surface-300 text-neuro-accent hover:text-neuro-accent-bright transition-colors"
                          title="Open Analysis Workspace"
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
                ))
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
              Are you sure you want to remove analysis session{' '}
              <strong className="text-neuro-text-primary font-mono">{deleteCandidateId}</strong> from this session? In demo mode, all changes are in-memory.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setDeleteCandidateId(null)}
                className="neuro-btn-secondary text-xs"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(deleteCandidateId)}
                className="neuro-btn-primary bg-neuro-error hover:bg-red-600 text-white text-xs"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
