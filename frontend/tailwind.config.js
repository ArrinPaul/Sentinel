/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        void: { 50: '#08090d', 100: '#0e1117', 200: '#161b26', 300: '#1c2233', 400: '#232b3e' },
        neon: { cyan: '#00f0ff', amber: '#ffb800', red: '#ff3366', green: '#00ff88', purple: '#a855f7' },
        ink: { 100: '#e8ecf4', 200: '#8896b0', 300: '#4a5568' },
      },
      fontFamily: {
        display: ['Syne', 'sans-serif'],
        body: ['Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 30px rgba(0,240,255,0.15), 0 0 60px rgba(0,240,255,0.05)',
        'glow-amber': '0 0 30px rgba(255,184,0,0.15), 0 0 60px rgba(255,184,0,0.05)',
        'glow-green': '0 0 30px rgba(0,255,136,0.15), 0 0 60px rgba(0,255,136,0.05)',
        'glow-red': '0 0 30px rgba(255,51,102,0.15), 0 0 60px rgba(255,51,102,0.05)',
        'glow-cyan-lg': '0 0 60px rgba(0,240,255,0.25), 0 0 120px rgba(0,240,255,0.1)',
      },
      animation: {
        'pulse-slow': 'pulse 4s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}
