/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#14213d", soft: "#3b4a68", mute: "#5b6472" },
        paper: { DEFAULT: "#f6f7f9", card: "#ffffff", line: "#dfe3e9" },
        accent: { DEFAULT: "#0f766e", dark: "#0b5a54", soft: "#e3f2f0" },
        real: { DEFAULT: "#1f7a46", soft: "#e4f4ea" },
        fake: { DEFAULT: "#b4301f", soft: "#fbe9e6" },
        warn: { DEFAULT: "#9a6200", soft: "#fdf1d8" },
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ['"Source Serif 4"', "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
