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
        'neuro-bg': 'rgb(var(--neuro-bg) / <alpha-value>)',
        'neuro-surface': {
          DEFAULT: 'rgb(var(--neuro-surface) / <alpha-value>)',
          200: 'rgb(var(--neuro-surface-200) / <alpha-value>)',
          300: 'rgb(var(--neuro-surface-300) / <alpha-value>)',
          400: 'rgb(var(--neuro-surface-400) / <alpha-value>)',
        },
        'neuro-accent': {
          DEFAULT: 'rgb(var(--neuro-accent) / <alpha-value>)',
          dim: 'rgb(var(--neuro-accent-dim) / <alpha-value>)',
          bright: 'rgb(var(--neuro-accent-bright) / <alpha-value>)',
          muted: 'rgb(var(--neuro-accent-muted) / <alpha-value>)',
        },
        'neuro-cyan': {
          DEFAULT: 'rgb(var(--neuro-cyan) / <alpha-value>)',
          dim: 'rgb(var(--neuro-cyan-dim) / <alpha-value>)',
          muted: 'rgb(var(--neuro-cyan-muted) / <alpha-value>)',
        },
        'neuro-blue': {
          DEFAULT: 'rgb(var(--neuro-blue) / <alpha-value>)',
          dim: 'rgb(var(--neuro-blue-dim) / <alpha-value>)',
          muted: 'rgb(var(--neuro-blue-muted) / <alpha-value>)',
        },
        'neuro-text': {
          primary: 'rgb(var(--neuro-text-primary) / <alpha-value>)',
          secondary: 'rgb(var(--neuro-text-secondary) / <alpha-value>)',
          muted: 'rgb(var(--neuro-text-muted) / <alpha-value>)',
          disabled: 'rgb(var(--neuro-text-disabled) / <alpha-value>)',
        },
        'neuro-border': {
          DEFAULT: 'rgb(var(--neuro-border) / <alpha-value>)',
          hover: 'rgb(var(--neuro-border-hover) / <alpha-value>)',
          active: 'rgb(var(--neuro-border-active) / <alpha-value>)',
        },
        'neuro-success': 'rgb(var(--neuro-success) / <alpha-value>)',
        'neuro-warning': 'rgb(var(--neuro-warning) / <alpha-value>)',
        'neuro-error': 'rgb(var(--neuro-error) / <alpha-value>)',
        'neuro-info': 'rgb(var(--neuro-info) / <alpha-value>)',
      },
      borderRadius: {
        sm: '6px',
        md: '8px',
        lg: '12px',
        xl: '16px',
      },
      boxShadow: {
        'neuro-sm': 'var(--neuro-shadow-sm)',
        'neuro-md': 'var(--neuro-shadow-md)',
        'neuro-lg': 'var(--neuro-shadow-lg)',
        'neuro-glow': 'var(--neuro-shadow-glow)',
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
