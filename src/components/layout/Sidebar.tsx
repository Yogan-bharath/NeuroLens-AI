import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  FilePlus2,
  FolderSearch,
  Sparkles,
  BarChart3,
  Sliders,
  Settings,
  FileText,
  ChevronLeft,
  ChevronRight,
  Brain,
  X,
} from 'lucide-react';
import { useApp } from '@/context/AppContext';
import { cn } from '@/lib/utils';

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Overview' },
  { path: '/new-analysis', icon: FilePlus2, label: 'New Analysis' },
  { path: '/model-selection', icon: Sliders, label: 'Model Selection' },
  { path: '/explainability', icon: Sparkles, label: 'Explainability Studio' },
  { path: '/history', icon: FolderSearch, label: 'Analysis History' },
  { path: '/model-performance', icon: BarChart3, label: 'Model Performance' },
  { path: '/settings', icon: Settings, label: 'Settings' },
  { path: '/docs', icon: FileText, label: 'Documentation' },
];

function SidebarContent({ collapsed, onClose }: { collapsed: boolean; onClose?: () => void }) {
  const location = useLocation();

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-neuro-border">
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-neuro-accent to-neuro-cyan flex items-center justify-center">
          <Brain className="w-5 h-5 text-neuro-bg" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: 'auto' }}
              exit={{ opacity: 0, width: 0 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <h1 className="text-base font-semibold text-neuro-text-primary">NeuroLens AI</h1>
            </motion.div>
          )}
        </AnimatePresence>
        {onClose && (
          <button
            onClick={onClose}
            className="ml-auto p-1 rounded hover:bg-neuro-surface-300 text-neuro-text-muted hover:text-neuro-text-primary lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto scrollbar-thin">
        {navItems.map(item => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onClose}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-200 group relative',
                isActive
                  ? 'bg-neuro-accent-muted text-neuro-accent-bright'
                  : 'text-neuro-text-secondary hover:text-neuro-text-primary hover:bg-neuro-surface-200'
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 bg-neuro-accent rounded-r-full"
                  transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                />
              )}
              <item.icon className={cn('w-5 h-5 flex-shrink-0', isActive ? 'text-neuro-accent' : 'text-neuro-text-muted group-hover:text-neuro-text-secondary')} />
              <AnimatePresence>
                {!collapsed && (
                  <motion.span
                    initial={{ opacity: 0, width: 0 }}
                    animate={{ opacity: 1, width: 'auto' }}
                    exit={{ opacity: 0, width: 0 }}
                    className="overflow-hidden whitespace-nowrap"
                  >
                    {item.label}
                  </motion.span>
                )}
              </AnimatePresence>
            </NavLink>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="border-t border-neuro-border p-3">
        {!collapsed && (
          <div className="px-2 py-2">
            <p className="text-xs text-neuro-text-muted">Research Platform</p>
            <p className="text-xs text-neuro-text-disabled mt-0.5">v0.1.0 — Demo Mode</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function Sidebar() {
  const { sidebarCollapsed, sidebarMobileOpen, toggleSidebar, setSidebarMobileOpen } = useApp();

  return (
    <>
      {/* Desktop sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: sidebarCollapsed ? 64 : 240 }}
        transition={{ duration: 0.2, ease: 'easeInOut' }}
        className="hidden lg:flex flex-col border-r border-neuro-border bg-neuro-surface h-screen sticky top-0 z-30 flex-shrink-0"
      >
        <SidebarContent collapsed={sidebarCollapsed} />
        <button
          onClick={toggleSidebar}
          className="absolute -right-3 top-20 z-40 w-6 h-6 rounded-full bg-neuro-surface-300 border border-neuro-border
                     flex items-center justify-center hover:bg-neuro-surface-400 transition-colors"
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? (
            <ChevronRight className="w-3.5 h-3.5 text-neuro-text-secondary" />
          ) : (
            <ChevronLeft className="w-3.5 h-3.5 text-neuro-text-secondary" />
          )}
        </button>
      </motion.aside>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {sidebarMobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/60 z-40 lg:hidden"
              onClick={() => setSidebarMobileOpen(false)}
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', damping: 25, stiffness: 250 }}
              className="fixed left-0 top-0 bottom-0 w-[280px] bg-neuro-surface border-r border-neuro-border z-50 lg:hidden"
            >
              <SidebarContent collapsed={false} onClose={() => setSidebarMobileOpen(false)} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
