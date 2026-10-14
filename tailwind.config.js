/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        'neuro-bg': '#0a0e1a',
        'neuro-surface': {
          DEFAULT: '#111827',
          200: '#1a2035',
          300: '#222b3d',
          400: '#2a3349',
        },
        'neuro-accent': {
          DEFAULT: '#14b8a6',
          dim: '#0d9488',
          bright: '#2dd4bf',
          muted: 'rgba(20, 184, 166, 0.15)',
        },
        'neuro-cyan': {
          DEFAULT: '#06b6d4',
          dim: '#0891b2',
          muted: 'rgba(6, 182, 212, 0.15)',
        },
        'neuro-blue': {
          DEFAULT: '#3b82f6',
          dim: '#2563eb',
          muted: 'rgba(59, 130, 246, 0.15)',
        },
        'neuro-text': {
          primary: '#f1f5f9',
          secondary: '#94a3b8',
          muted: '#64748b',
          disabled: '#475569',
        },
        'neuro-border': {
          DEFAULT: '#1e293b',
          hover: '#334155',
          active: '#475569',
        },
        'neuro-success': '#22c55e',
        'neuro-warning': '#f59e0b',
        'neuro-error': '#ef4444',
        'neuro-info': '#3b82f6',
      },
      borderRadius: {
        sm: '6px',
        md: '8px',
        lg: '12px',
        xl: '16px',
      },
      boxShadow: {
        'neuro-sm': '0 1px 3px rgba(0, 0, 0, 0.3), 0 1px 2px rgba(0, 0, 0, 0.2)',
        'neuro-md': '0 4px 6px rgba(0, 0, 0, 0.3), 0 2px 4px rgba(0, 0, 0, 0.2)',
        'neuro-lg': '0 10px 15px rgba(0, 0, 0, 0.3), 0 4px 6px rgba(0, 0, 0, 0.2)',
        'neuro-glow': '0 0 20px rgba(20, 184, 166, 0.15)',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-in-right': 'slideInRight 0.3s ease-out',
        'slide-in-up': 'slideInUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan-line': 'scanLine 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(20px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scanLine: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
};
