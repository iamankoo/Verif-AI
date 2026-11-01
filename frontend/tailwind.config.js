/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Clean light theme: slate text on white surfaces, one teal accent.
        ink: { DEFAULT: "#0f172a", soft: "#334155", mute: "#64748b" },
        paper: { DEFAULT: "#f8fafc", card: "#ffffff", raised: "#f1f5f9", line: "#e2e8f0" },
        accent: { DEFAULT: "#0f766e", dark: "#115e59", soft: "#e6f4f1" },
        real: { DEFAULT: "#15803d", soft: "#dcfce7" },
        fake: { DEFAULT: "#b91c1c", soft: "#fee2e2" },
        warn: { DEFAULT: "#b45309", soft: "#fef3c7" },
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
