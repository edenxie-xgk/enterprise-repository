/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          primary: '#165DFF',
          secondary: '#F5F7FA',
          accent: '#4080FF',
          dark: '#1D2129',
          light: '#F9FAFC'
        },
        fontFamily: {
          inter: ['Inter', 'system-ui', 'sans-serif'],
        },
      },
    },
    plugins: [],
  }