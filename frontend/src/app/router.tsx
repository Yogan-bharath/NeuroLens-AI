import { createBrowserRouter, Navigate } from 'react-router-dom';
import AppLayout from '@/components/layout/AppLayout';
import OverviewPage from '@/pages/OverviewPage';
import NewAnalysisPage from '@/pages/NewAnalysisPage';
import ModelSelectionPage from '@/pages/ModelSelectionPage';
import AnalysisWorkspacePage from '@/pages/AnalysisWorkspacePage';
import Classification2DResultsPage from '@/pages/Classification2DResultsPage';
import Segmentation3DResultsPage from '@/pages/Segmentation3DResultsPage';
import CombinedResultsPage from '@/pages/CombinedResultsPage';
import ExplainabilityPage from '@/pages/ExplainabilityPage';
import HistoryPage from '@/pages/HistoryPage';
import ModelPerformancePage from '@/pages/ModelPerformancePage';
import SettingsPage from '@/pages/SettingsPage';
import DocumentationPage from '@/pages/DocumentationPage';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <OverviewPage />,
      },
      {
        path: 'new-analysis',
        element: <NewAnalysisPage />,
      },
      {
        path: 'model-selection',
        element: <ModelSelectionPage />,
      },
      {
        path: 'analysis/:id',
        element: <AnalysisWorkspacePage />,
      },
      {
        path: 'analysis/:analysisId/2d-results',
        element: <Classification2DResultsPage />,
      },
      {
        path: 'analysis/:analysisId/3d-results',
        element: <Segmentation3DResultsPage />,
      },
      {
        path: 'analysis/:analysisId/combined-results',
        element: <CombinedResultsPage />,
      },
      {
        path: 'explainability',
        element: <ExplainabilityPage />,
      },
      {
        path: 'history',
        element: <HistoryPage />,
      },
      {
        path: 'model-performance',
        element: <ModelPerformancePage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
      {
        path: 'docs',
        element: <DocumentationPage />,
      },
      {
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);
