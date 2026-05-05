/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{jsx,js,tsx,ts}'],
  darkMode: 'class', // We will use a class to toggle dark mode
  theme: {
    extend: {
      colors: {
        brand: {
          sage: '#c5d39e',
          sky: '#60a5fa',
          ochre: '#f59e0b',
          dark: '#0a0a0a',
          light: '#fbfbfb',
        }
      },
      fontFamily: {
        heading: ['Geist', 'sans-serif'],
        body: ['Satoshi', 'sans-serif'],
      },
    },
  },
  plugins: [],
}