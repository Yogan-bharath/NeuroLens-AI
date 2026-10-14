import type { AnalysisResult, AnalysisSummary, ExplanationResult, SegmentationResult } from '@/types/analysis';
import type { ModelMetrics } from '@/types/model';
import type { ChatMessage } from '@/types/chat';
import * as mockApi from './mockApi';
import * as chatService from './chatService';

// Backend configuration flag
const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || null;
export const isUsingMockBackend = !API_BASE_URL;

export async function analyzeMRI(imageDataUrl: string, fileName: string): Promise<AnalysisResult> {
  if (isUsingMockBackend) {
    return mockApi.mockAnalyzeMRI(imageDataUrl, fileName);
  }

  // Future FastAPI integration point
  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: imageDataUrl, fileName }),
  });

  if (!response.ok) {
    throw new Error(`Inference API error: ${response.statusText}`);
  }

  return response.json();
}

export async function getAnalysis(analysisId: string): Promise<AnalysisResult | null> {
  if (isUsingMockBackend) {
    return mockApi.mockGetAnalysis(analysisId);
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/analysis/${analysisId}`);
  if (!response.ok) return null;
  return response.json();
}

export async function getAnalysisHistory(): Promise<AnalysisSummary[]> {
  if (isUsingMockBackend) {
    return mockApi.mockGetAnalysisHistory();
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/history`);
  if (!response.ok) return [];
  return response.json();
}

export async function getModelMetrics(): Promise<ModelMetrics> {
  if (isUsingMockBackend) {
    return mockApi.mockGetModelMetrics();
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/model/metrics`);
  if (!response.ok) throw new Error('Failed to fetch model metrics');
  return response.json();
}

export async function getExplanation(
  analysisId: string,
  method: 'gradcam' | 'gradcam_pp'
): Promise<ExplanationResult | null> {
  const analysis = await getAnalysis(analysisId);
  return analysis?.explanations.find(e => e.method === method) || null;
}

export async function getSegmentation(analysisId: string): Promise<SegmentationResult | null> {
  const analysis = await getAnalysis(analysisId);
  return analysis?.segmentation || null;
}

export async function sendChatMessage(message: string): Promise<ChatMessage> {
  if (isUsingMockBackend) {
    return chatService.mockSendChatMessage(message);
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });

  return response.json();
}
