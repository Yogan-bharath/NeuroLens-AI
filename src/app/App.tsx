import { RouterProvider } from 'react-router-dom';
import { router } from './router';
import { AppProvider } from '@/context/AppContext';
import { AnalysisProvider } from '@/context/AnalysisContext';
import { ConfigurationProvider } from '@/context/ConfigurationContext';

export default function App() {
  return (
    <AppProvider>
      <ConfigurationProvider>
        <AnalysisProvider>
          <RouterProvider router={router} />
        </AnalysisProvider>
      </ConfigurationProvider>
    </AppProvider>
  );
}
