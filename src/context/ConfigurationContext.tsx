import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import type {
  AnalysisConfiguration,
  AnalysisMode,
  ClassificationModel,
  SegmentationModel,
  XaiMethod,
  FeatureLayerOption,
} from '@/types/modelRegistry';
import {
  classificationModelsRegistry,
  segmentationModelsRegistry,
  xaiMethodsRegistry,
  validateConfiguration,
  type ValidationResult,
} from '@/services/modelRegistry';

interface ConfigurationContextState {
  config: AnalysisConfiguration;
  setMode: (mode: AnalysisMode) => void;
  setClassificationModel: (modelId: string) => void;
  setSegmentationModel: (modelId: string) => void;
  setClassificationXaiMethod: (methodId: string) => void;
  setClassificationLayer: (layerId: string) => void;
  setSegmentationXaiMethod: (methodId: string) => void;
  setSegmentationLayer: (layerId: string) => void;
  setSegmentationVisualization: (vis: 'mask' | 'overlay' | 'comparison') => void;
  setSegmentationOpacity: (opacity: number) => void;
  currentClassificationModel: ClassificationModel | undefined;
  currentSegmentationModel: SegmentationModel | undefined;
  currentXaiMethod: XaiMethod | undefined;
  currentSegmentationXaiMethod: XaiMethod | undefined;
  availableLayers: FeatureLayerOption[];
  availableSegmentationLayers: FeatureLayerOption[];
  supportedXaiMethods: XaiMethod[];
  supportedSegmentationXaiMethods: XaiMethod[];
  validation: ValidationResult;
  compatibilityMessage: string | null;
  clearCompatibilityMessage: () => void;
  resetToDefaults: () => void;
}

const defaultConfig: AnalysisConfiguration = {
  mode: 'combined',
  classificationModelId: 'efficientnetv2b0',
  segmentationModelId: 'unet',
  classificationXaiMethod: 'gradcam',
  classificationTarget: 'Predicted Top Class',
  classificationLayer: 'top_conv',
  segmentationXaiMethod: 'seg_gradcam',
  segmentationLayer: 'bottleneck.conv2',
  segmentationVisualization: 'comparison', // Default is permanently side-by-side
  segmentationOpacity: 0.7,
};

const ConfigurationContext = createContext<ConfigurationContextState | null>(null);

export function ConfigurationProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<AnalysisConfiguration>(defaultConfig);
  const [compatibilityMessage, setCompatibilityMessage] = useState<string | null>(null);

  const currentClassificationModel = classificationModelsRegistry.find(
    m => m.id === config.classificationModelId
  );
  const currentSegmentationModel = segmentationModelsRegistry.find(
    m => m.id === config.segmentationModelId
  );
  const currentXaiMethod = xaiMethodsRegistry.find(
    x => x.id === config.classificationXaiMethod
  );
  const currentSegmentationXaiMethod = xaiMethodsRegistry.find(
    x => x.id === config.segmentationXaiMethod
  );

  const availableLayers = currentClassificationModel?.compatibleLayers || [];
  const availableSegmentationLayers = currentSegmentationModel?.compatibleLayers || [];

  const supportedXaiMethods = xaiMethodsRegistry.filter(x =>
    x.task === 'classification' && currentClassificationModel?.supportedXaiMethods.includes(x.id)
  );
  const supportedSegmentationXaiMethods = xaiMethodsRegistry.filter(x =>
    x.task === 'segmentation' && currentSegmentationModel?.supportedXaiMethods.includes(x.id)
  );

  const validation = validateConfiguration(config);

  const setMode = useCallback((mode: AnalysisMode) => {
    setConfig(prev => ({ ...prev, mode }));
  }, []);

  const setClassificationModel = useCallback((modelId: string) => {
    const targetModel = classificationModelsRegistry.find(m => m.id === modelId);
    if (!targetModel) return;

    setConfig(prev => {
      let nextXai = prev.classificationXaiMethod;
      let nextLayer = prev.classificationLayer;
      let message: string | null = null;

      if (!targetModel.supportedXaiMethods.includes(prev.classificationXaiMethod)) {
        nextXai = targetModel.supportedXaiMethods[0] || 'gradcam';
        const oldXaiName = xaiMethodsRegistry.find(x => x.id === prev.classificationXaiMethod)?.name || prev.classificationXaiMethod;
        const newXaiName = xaiMethodsRegistry.find(x => x.id === nextXai)?.name || nextXai;
        message = `${oldXaiName} is not currently supported by ${targetModel.name}. Switched default to ${newXaiName}.`;
      }

      const defaultLayer = targetModel.compatibleLayers.find(l => l.isVerified)?.id || targetModel.compatibleLayers[0]?.id || 'default';
      nextLayer = defaultLayer;

      if (message) {
        setCompatibilityMessage(message);
      } else {
        setCompatibilityMessage(null);
      }

      return {
        ...prev,
        classificationModelId: modelId,
        classificationXaiMethod: nextXai,
        classificationLayer: nextLayer,
      };
    });
  }, []);

  const setSegmentationModel = useCallback((modelId: string) => {
    const targetModel = segmentationModelsRegistry.find(m => m.id === modelId);
    if (!targetModel) return;

    setConfig(prev => {
      let nextXai = prev.segmentationXaiMethod;
      let nextLayer = prev.segmentationLayer;

      if (!targetModel.supportedXaiMethods.includes(prev.segmentationXaiMethod)) {
        nextXai = targetModel.supportedXaiMethods[0] || 'seg_gradcam';
      }

      const defaultLayer = targetModel.compatibleLayers.find(l => l.isVerified)?.id || targetModel.compatibleLayers[0]?.id || 'bottleneck.conv2';
      nextLayer = defaultLayer;

      return {
        ...prev,
        segmentationModelId: modelId,
        segmentationXaiMethod: nextXai,
        segmentationLayer: nextLayer,
      };
    });
  }, []);

  const setClassificationXaiMethod = useCallback((methodId: string) => {
    setConfig(prev => ({
      ...prev,
      classificationXaiMethod: methodId,
    }));
  }, []);

  const setClassificationLayer = useCallback((layerId: string) => {
    setConfig(prev => ({ ...prev, classificationLayer: layerId }));
  }, []);

  const setSegmentationXaiMethod = useCallback((methodId: string) => {
    setConfig(prev => ({ ...prev, segmentationXaiMethod: methodId }));
  }, []);

  const setSegmentationLayer = useCallback((layerId: string) => {
    setConfig(prev => ({ ...prev, segmentationLayer: layerId }));
  }, []);

  const setSegmentationVisualization = useCallback((vis: 'mask' | 'overlay' | 'comparison') => {
    setConfig(prev => ({ ...prev, segmentationVisualization: vis }));
  }, []);

  const setSegmentationOpacity = useCallback((opacity: number) => {
    setConfig(prev => ({ ...prev, segmentationOpacity: opacity }));
  }, []);

  const clearCompatibilityMessage = useCallback(() => {
    setCompatibilityMessage(null);
  }, []);

  const resetToDefaults = useCallback(() => {
    setConfig(defaultConfig);
    setCompatibilityMessage(null);
  }, []);

  return (
    <ConfigurationContext.Provider
      value={{
        config,
        setMode,
        setClassificationModel,
        setSegmentationModel,
        setClassificationXaiMethod,
        setClassificationLayer,
        setSegmentationXaiMethod,
        setSegmentationLayer,
        setSegmentationVisualization,
        setSegmentationOpacity,
        currentClassificationModel,
        currentSegmentationModel,
        currentXaiMethod,
        currentSegmentationXaiMethod,
        availableLayers,
        availableSegmentationLayers,
        supportedXaiMethods,
        supportedSegmentationXaiMethods,
        validation,
        compatibilityMessage,
        clearCompatibilityMessage,
        resetToDefaults,
      }}
    >
      {children}
    </ConfigurationContext.Provider>
  );
}

export function useConfiguration() {
  const context = useContext(ConfigurationContext);
  if (!context) throw new Error('useConfiguration must be used within ConfigurationProvider');
  return context;
}
