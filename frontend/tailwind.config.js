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
          'dark-plum': '#1A1A1D',
          'dark-berry': '#3B1C32',
          'dark-magenta': '#6A1E55',
          'dark-pink': '#A64D79',
          'light-hotpink': '#F13E93',
          'light-pink': '#F891BB',
          'light-peach': '#F9D0CD',
          'light-yellow': '#FAFFCB'
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