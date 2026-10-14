import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';

interface AppState {
  sidebarCollapsed: boolean;
  sidebarMobileOpen: boolean;
  isDemoMode: boolean;
  toggleSidebar: () => void;
  setSidebarMobileOpen: (open: boolean) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarMobileOpen, setSidebarMobileOpen] = useState(false);

  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed(prev => !prev);
  }, []);

  return (
    <AppContext.Provider
      value={{
        sidebarCollapsed,
        sidebarMobileOpen,
        isDemoMode: true,
        toggleSidebar,
        setSidebarMobileOpen,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) throw new Error('useApp must be used within AppProvider');
  return context;
}
