/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-bg': '#0f0f1a',
        'dark-surface': '#1a1a2e',
        'dark-card': '#252542',
        'dark-border': '#3d3d5c',
        'cyber-blue': '#00d4ff',
        'cyber-purple': '#a855f7',
        'cyber-pink': '#ec4899',
        'cyber-green': '#22c55e',
        'cyber-yellow': '#eab308',
        'cyber-red': '#ef4444',
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'cyber': '0 4px 30px rgba(0, 212, 255, 0.1)',
        'glow-blue': '0 0 20px rgba(0, 212, 255, 0.3)',
        'glow-purple': '0 0 20px rgba(168, 85, 247, 0.3)',
      },
    },
  },
  plugins: [],
}
