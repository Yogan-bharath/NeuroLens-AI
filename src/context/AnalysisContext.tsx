import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type { AnalysisResult } from '@/types/analysis';
import type { AnalysisConfiguration } from '@/types/modelRegistry';
import { mockAnalyzeMRI, mockGetAnalysis, addAnalysisToHistory } from '@/services/mockApi';

interface AnalysisContextState {
  currentAnalysis: AnalysisResult | null;
  analysisStatus: string;
  isAnalyzing: boolean;
  statusStage: number;
  runAnalysis: (imageDataUrl: string, fileName: string, config?: AnalysisConfiguration) => Promise<AnalysisResult>;
  loadAnalysis: (id: string) => Promise<void>;
  setCurrentAnalysis: (analysis: AnalysisResult | null) => void;
  clearAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextState | null>(null);

const stages = [
  'Uploading image...',
  'Preprocessing scan...',
  'Running classification...',
  'Generating explanations...',
  'Analysis complete',
];

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [currentAnalysis, setCurrentAnalysis] = useState<AnalysisResult | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [statusStage, setStatusStage] = useState(0);

  const runAnalysis = useCallback(async (
    imageDataUrl: string,
    fileName: string,
    config?: AnalysisConfiguration
  ): Promise<AnalysisResult> => {
    setIsAnalyzing(true);
    setStatusStage(0);
    setAnalysisStatus(stages[0]);

    // Simulate stages
    const stageInterval = setInterval(() => {
      setStatusStage(prev => {
        const next = Math.min(prev + 1, stages.length - 1);
        setAnalysisStatus(stages[next]);
        return next;
      });
    }, 900);

    try {
      const result = await mockAnalyzeMRI(imageDataUrl, fileName, config);
      clearInterval(stageInterval);
      setStatusStage(stages.length - 1);
      setAnalysisStatus(stages[stages.length - 1]);
      setCurrentAnalysis(result);
      addAnalysisToHistory(result);
      return result;
    } catch {
      clearInterval(stageInterval);
      setAnalysisStatus('Analysis failed');
      throw new Error('Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const loadAnalysis = useCallback(async (id: string) => {
    const result = await mockGetAnalysis(id);
    if (result) {
      setCurrentAnalysis(result);
    }
  }, []);

  const clearAnalysis = useCallback(() => {
    setCurrentAnalysis(null);
    setAnalysisStatus('');
    setStatusStage(0);
  }, []);

  return (
    <AnalysisContext.Provider
      value={{
        currentAnalysis,
        analysisStatus,
        isAnalyzing,
        statusStage,
        runAnalysis,
        loadAnalysis,
        setCurrentAnalysis,
        clearAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (!context) throw new Error('useAnalysis must be used within AnalysisProvider');
  return context;
}
