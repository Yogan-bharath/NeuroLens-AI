import type { AnalysisResult, AnalysisSummary, ExplanationResult, SegmentationResult, ActualModelMetadata } from '@/types/analysis';
import type {
  ModelMetrics,
  ClassificationPerformanceResponse,
  SegmentationPerformanceResponse,
  UnifiedModelOverview,
  ModelComparisonMatrix,
  EvaluationHistoryItem,
} from '@/types/model';
import type { ChatMessage } from '@/types/chat';
import type { AnalysisConfiguration } from '@/types/modelRegistry';
import type { User, AuthResponse, LoginCredentials, RegisterCredentials } from '@/types/auth';
import * as mockApi from './mockApi';
import * as chatService from './chatService';

// Backend configuration flag
const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || null;
export const isUsingMockBackend = !API_BASE_URL;

// ─── Authentication Token Storage Helpers ─────────────────────────────────
export function getStoredToken(): string | null {
  return localStorage.getItem('neurolens_token');
}

export function setStoredToken(token: string): void {
  localStorage.setItem('neurolens_token', token);
}

export function removeStoredToken(): void {
  localStorage.removeItem('neurolens_token');
}

export function getAuthHeaders(): HeadersInit {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

// ─── MRI Analysis & Inference Service Calls ───────────────────────────────
export async function analyzeMRI(
  imageDataUrl: string,
  fileName: string,
  config?: AnalysisConfiguration
): Promise<AnalysisResult> {
  if (isUsingMockBackend) {
    return mockApi.mockAnalyzeMRI(imageDataUrl, fileName, config);
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ image: imageDataUrl, fileName, config }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Inference API error: ${response.statusText}`);
  }

  return response.json();
}

export interface BratsSegmentationResponse {
  result: Blob;
  metadata: Record<string, unknown>;
}

export interface SegmentationStatusResponse {
  checkpoint_exists: boolean;
  checkpoint_path: string;
  architecture_initialized: boolean;
  tensorflow_version: string;
  gpu_available: boolean;
  gpu_count: number;
  segmentation_ready: boolean;
  model_details: {
    name: string;
    framework: string;
    patch_size: number[];
    stride: number[];
    modalities: string[];
    classes: Array<{ index: number; name: string; source_label: number; color: string }>;
    target_xai_layer: string;
  };
  status_reason?: string | null;
}

export interface SegmentationAnalysisResponse {
  success: boolean;
  analysis_id?: string;
  prediction_id: string;
  download_url: string;
  model_id?: string;
  model_name?: string;
  model_version?: string;
  actual_model?: ActualModelMetadata;
  volume_dimensions: [number, number, number];
  voxel_spacing: [number, number, number];
  anatomical_orientation: string[];
  processing_time_ms: number;
  class_distribution: Array<{
    index: number;
    name: string;
    voxel_count: number;
    percentage: number;
    volume_mm3: number;
    color: string;
  }>;
  slices: {
    axial: { slice_index: number; preview: string };
    coronal: { slice_index: number; preview: string };
    sagittal: { slice_index: number; preview: string };
  };
  surface_3d: {
    bounding_box: { min: number[]; max: number[]; center: number[] };
    classes: Array<{ class_idx: number; name: string; hex: string }>;
    points: Record<string, number[][]>;
    brain_surface?: number[][];
    anatomical_location?: {
      hemisphere: string;
      hemisphere_detail: string;
      anterior_posterior: string;
      ap_detail: string;
      superior_inferior: string;
      si_detail: string;
      summary: string;
      center_x_mm: number;
      center_y_mm: number;
      center_z_mm: number;
    };
  };
  evaluation_metrics?: Record<string, number> | null;
  explainability: {
    gradcam_layer?: string;
    gradcam_maps?: Record<string, string>;
    bottleneck_features?: {
      shape: number[];
      channels_count: number;
      depth_slice: number;
    };
    modality_ablation?: Array<{
      removed_modality: string;
      tumor_voxel_count: number;
      relative_change_pct: number;
      impact_level: string;
    }>;
    uncertainty?: {
      mean_entropy_overall: number;
      mean_entropy_tumor: number;
    };
    error?: string;
  };
  disclaimer: string;
}

export interface ClassificationAnalysisResponse {
  success: boolean;
  analysis_id?: string;
  prediction_id: string;
  model_id: string;
  model_name: string;
  model_version?: string;
  actual_model?: ActualModelMetadata;
  top_prediction: string;
  confidence: number;
  scores: Array<{
    label: string;
    score: number;
    percentage: number;
    isTopPrediction: boolean;
  }>;
  gradcam_overlay?: string | null;
  processing_time_ms: number;
  disclaimer: string;
}

export interface CombinedAnalysisResponse {
  success: boolean;
  combined_analysis_id: string;
  analysis_id: string;
  classification_id?: string | null;
  segmentation_id?: string | null;
  classification_status: 'completed' | 'failed';
  segmentation_status: 'completed' | 'failed';
  status: 'completed' | 'partial' | 'failed';
  classification?: any;
  segmentation?: any;
  errors: {
    classification?: string | null;
    segmentation?: string | null;
  };
}

export async function getSegmentationStatus(): Promise<SegmentationStatusResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/segmentation/status` : '/api/segmentation/status';
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) {
    throw new Error('Failed to retrieve segmentation status.');
  }
  return response.json();
}

export interface StudyValidationReport {
  is_valid: boolean;
  status: 'ready' | 'action_required';
  summary: string;
  modalities_detected: string[];
  missing_modalities: string[];
  volume_dimensions: [number, number, number] | null;
  voxel_spacing: [number, number, number] | null;
  anatomical_orientation: string[] | null;
  spatial_alignment: {
    is_aligned: boolean;
    reference_modality?: string;
  };
  modality_reports: Record<string, {
    filename: string;
    modality: string;
    dimensions?: number[];
    voxel_spacing?: number[];
    orientation?: string[];
    size_bytes?: number;
    is_readable: boolean;
    is_finite?: boolean;
    min_intensity?: number;
    max_intensity?: number;
    aligned_with_reference?: boolean;
    error?: string;
  }>;
  errors: string[];
  warnings: string[];
}

export async function validateSegmentationStudy(formData: FormData): Promise<StudyValidationReport> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/segmentation/validate-study` : '/api/segmentation/validate-study';
  const token = getStoredToken();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Study validation failed with status ${response.status}`);
  }

  return response.json();
}

export async function analyzeSegmentationVolume(formData: FormData): Promise<SegmentationAnalysisResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/segmentation/analyze` : '/api/segmentation/analyze';
  const token = getStoredToken();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `3D Segmentation failed with status ${response.status}`);
  }
  return response.json();
}

export async function analyzeClassification2D(
  file: File,
  modelId: string = 'efficientnetv2b0',
  useClahe?: boolean
): Promise<ClassificationAnalysisResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/classification/analyze` : '/api/classification/analyze';
  const token = getStoredToken();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  const formData = new FormData();
  formData.append('image', file);
  formData.append('model_id', modelId);
  if (useClahe !== undefined) {
    formData.append('use_clahe', String(useClahe));
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Classification failed with status ${response.status}`);
  }
  return response.json();
}

export async function analyzeCombinedWorkflows(formData: FormData): Promise<CombinedAnalysisResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/analyses/combined` : '/api/analyses/combined';
  const token = getStoredToken();
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Combined analysis failed with status ${response.status}`);
  }
  return response.json();
}

export async function segmentBratsVolume(
  files: { flair: File; t1: File; t1ce: File; t2: File; segmentation?: File }
): Promise<BratsSegmentationResponse> {
  if (!API_BASE_URL) {
    throw new Error('Volumetric segmentation requires the configured NeuroLens backend.');
  }

  const formData = new FormData();
  formData.append('flair', files.flair);
  formData.append('t1', files.t1);
  formData.append('t1ce', files.t1ce);
  formData.append('t2', files.t2);
  if (files.segmentation) {
    formData.append('segmentation', files.segmentation);
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/segment/brats`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'BraTS Segmentation request failed.');
  }

  const resultBlob = await response.blob();
  const metadataHeader = response.headers.get('X-Analysis-Metadata');
  let metadata: Record<string, unknown> = {};

  if (metadataHeader) {
    try {
      metadata = JSON.parse(metadataHeader);
    } catch (e) {
      console.warn('Failed to parse metadata header from BraTS segmentation:', e);
    }
  }

  return {
    result: resultBlob,
    metadata,
  };
}

export async function getAnalysis(analysisId: string): Promise<AnalysisResult | null> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/analyses/${analysisId}` : `/api/analyses/${analysisId}`;
  try {
    const response = await fetch(url, { headers: getAuthHeaders() });
    if (response.ok) return response.json();
  } catch (err) {
    console.warn(`Failed to fetch analysis ${analysisId}:`, err);
  }
  if (isUsingMockBackend) {
    return mockApi.mockGetAnalysis(analysisId);
  }
  return null;
}

export async function getAnalysisHistory(): Promise<AnalysisSummary[]> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/history` : '/api/history';
  try {
    const response = await fetch(url, { headers: getAuthHeaders() });
    if (response.ok) {
      const data = await response.json();
      return Array.isArray(data) ? data : (data.items || []);
    }
  } catch (err) {
    console.warn('Failed to fetch live analysis history:', err);
  }
  if (isUsingMockBackend) {
    return mockApi.mockGetAnalysisHistory();
  }
  return [];
}

export async function deleteAnalysis(analysisId: string): Promise<boolean> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/history/${analysisId}` : `/api/history/${analysisId}`;
  const response = await fetch(url, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (isUsingMockBackend) {
      mockApi.mockDeleteAnalysis(analysisId);
      return true;
    }
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to delete analysis session.');
  }

  return true;
}

export async function getModelMetrics(): Promise<ModelMetrics> {
  if (isUsingMockBackend) {
    return mockApi.mockGetModelMetrics();
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/metrics`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) return mockApi.mockGetModelMetrics();
  return response.json();
}

export async function getClassificationPerformance(): Promise<ClassificationPerformanceResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/performance/classification` : '/api/performance/classification';
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) throw new Error('Failed to fetch classification benchmark metrics');
  return response.json();
}

export async function getSegmentationPerformance(): Promise<SegmentationPerformanceResponse> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/performance/segmentation` : '/api/performance/segmentation';
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) throw new Error('Failed to fetch segmentation benchmark metrics');
  return response.json();
}

export async function getAllModelsOverview(): Promise<UnifiedModelOverview[]> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/performance/models` : '/api/performance/models';
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) throw new Error('Failed to fetch unified model overview');
  return response.json();
}

export async function getModelComparison(): Promise<ModelComparisonMatrix> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/performance/compare` : '/api/performance/compare';
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) throw new Error('Failed to fetch model comparison matrix');
  return response.json();
}

export async function getEvaluationHistory(): Promise<EvaluationHistoryItem[]> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/performance/evaluation-history` : '/api/performance/evaluation-history';
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) throw new Error('Failed to fetch evaluation history');
  return response.json();
}

export async function getAnalysisById(analysisId: string): Promise<any> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/analyses/${analysisId}` : `/api/analyses/${analysisId}`;
  const response = await fetch(url, { headers: getAuthHeaders() });
  if (!response.ok) throw new Error(`Analysis not found: ${response.statusText}`);
  return response.json();
}

export async function getExplanation(
  analysisId: string,
  method: string = 'gradcam'
): Promise<ExplanationResult | null> {
  const analysis = await getAnalysis(analysisId);
  if (!analysis) return null;
  return analysis.explanations.find(e => e.method === method) || null;
}

export async function generateExplanation(
  analysisId: string,
  method: string,
  targetLayer?: string,
  targetClass?: string
): Promise<any> {
  const url = API_BASE_URL ? `${API_BASE_URL}/api/analyses/${analysisId}/explain` : `/api/analyses/${analysisId}/explain`;
  const response = await fetch(url, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      method,
      target_layer: targetLayer,
      target_class: targetClass,
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to generate explanation.');
  }
  return response.json();
}

export async function getSegmentation(analysisId: string): Promise<SegmentationResult | null> {
  const analysis = await getAnalysis(analysisId);
  return analysis?.segmentation || null;
}

// ─── Research Copilot Chat Endpoints ─────────────────────────────────────
export interface CopilotResponse {
  reply: string;
  conversationId: string;
  sources?: string[];
}

export async function sendChatMessage(message: string, analysisId?: string, conversationId?: string): Promise<CopilotResponse> {
  if (isUsingMockBackend) {
    const res = await chatService.mockSendChatMessage(message);
    return {
      reply: res.content,
      conversationId: conversationId || res.id,
    };
  }

  try {
    const response = await fetch(`${API_BASE_URL}/api/copilot/chat`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ message, analysis_id: analysisId, conversation_id: conversationId }),
    });
    if (response.ok) {
      const data = await response.json();
      return {
        reply: data.reply || data.response || '',
        conversationId: data.conversation_id || data.conversationId || conversationId || 'conv-1',
        sources: data.sources,
      };
    }
  } catch (err) {
    console.warn('Live copilot unavailable, falling back to local assistant:', err);
  }

  const res = await chatService.mockSendChatMessage(message);
  return {
    reply: res.content,
    conversationId: conversationId || res.id,
  };
}

export async function clearCopilotConversation(conversationId: string): Promise<void> {
  if (isUsingMockBackend) return;
  await fetch(`${API_BASE_URL}/api/copilot/clear-history`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ conversation_id: conversationId }),
  });
}

// ─── Authentication Endpoints ─────────────────────────────────────────────
export async function loginUser(credentials: LoginCredentials): Promise<AuthResponse> {
  if (isUsingMockBackend) {
    const mockUser: User = {
      id: 'usr_demo',
      name: 'Dr. Sarah Lin, MD',
      email: credentials.email,
      role: 'researcher',
      is_active: true,
      created_at: new Date().toISOString(),
    };
    const authData: AuthResponse = {
      access_token: 'mock_jwt_token',
      token_type: 'bearer',
      expires_in: 86400,
      user: mockUser,
    };
    setStoredToken(authData.access_token);
    return authData;
  }

  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Invalid email or password.');
  }

  const authData: AuthResponse = await response.json();
  setStoredToken(authData.access_token);
  return authData;
}

export async function registerUser(credentials: RegisterCredentials): Promise<AuthResponse> {
  if (isUsingMockBackend) {
    const mockUser: User = {
      id: 'usr_new',
      name: credentials.name,
      email: credentials.email,
      role: credentials.role || 'researcher',
      is_active: true,
      created_at: new Date().toISOString(),
    };
    return {
      access_token: 'mock_jwt_token',
      token_type: 'bearer',
      expires_in: 86400,
      user: mockUser,
    };
  }

  const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Registration failed.');
  }

  return response.json();
}

export async function fetchCurrentUser(): Promise<User | null> {
  const token = getStoredToken();
  if (!token && !isUsingMockBackend) return null;

  if (isUsingMockBackend) {
    return {
      id: 'usr_demo',
      name: 'Dr. Sarah Lin, MD',
      email: 'clinician@neurolens.ai',
      role: 'researcher',
      is_active: true,
      created_at: new Date().toISOString(),
    };
  }

  const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    removeStoredToken();
    return null;
  }

  return response.json();
}

export async function logoutUser(): Promise<void> {
  removeStoredToken();
  if (!isUsingMockBackend) {
    await fetch(`${API_BASE_URL}/api/auth/logout`, {
      method: 'POST',
    }).catch(() => {});
  }
}

export async function generate2DSegmentation(analysisId: string): Promise<{
  success: boolean;
  analysis_id: string;
  segmentation_mask_url: string;
  segmentation_overlay_url: string;
  segmentation_model_used: string;
  segmentation_tumor_detected: boolean;
  segmentation_tumor_area_pct: number;
}> {
  if (isUsingMockBackend) {
    throw new Error('2D segmentation requires live backend connection.');
  }

  const response = await fetch(`${API_BASE_URL}/api/classification/${analysisId}/segment`, {
    method: 'POST',
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || '2D tumor segmentation failed.');
  }

  return response.json();
}
