import { RouterProvider } from 'react-router-dom';
import { router } from './router';
import { AppProvider } from '@/context/AppContext';
import { AuthProvider } from '@/context/AuthContext';
import { AnalysisProvider } from '@/context/AnalysisContext';
import { ConfigurationProvider } from '@/context/ConfigurationContext';
import AuthModal from '@/components/auth/AuthModal';

export default function App() {
  return (
    <AppProvider>
      <AuthProvider>
        <ConfigurationProvider>
          <AnalysisProvider>
            <RouterProvider router={router} />
            <AuthModal />
          </AnalysisProvider>
        </ConfigurationProvider>
      </AuthProvider>
    </AppProvider>
  );
}
