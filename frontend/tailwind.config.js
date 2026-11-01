/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Dark navy / deep teal forensic theme. Text tokens are light, surface tokens are dark.
        ink: { DEFAULT: "#f2fbfb", soft: "#bfdcdd", mute: "#82a8ac" },
        paper: { DEFAULT: "#04121b", card: "#071f29", raised: "#0a2c36", line: "#14474d" },
        accent: { DEFAULT: "#1fe3c4", dark: "#14b9a1", soft: "#0a3a3d" },
        real: { DEFAULT: "#34d57f", soft: "#0c3324" },
        fake: { DEFAULT: "#ff6262", soft: "#3a1519" },
        warn: { DEFAULT: "#f5b942", soft: "#3a2c0d" },
      },
      fontFamily: {
        sans: ['"Inter Variable"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(31,227,196,0.30), 0 10px 50px -14px rgba(31,227,196,0.35)",
        card: "0 1px 0 0 rgba(31,227,196,0.08) inset, 0 8px 30px -12px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};
