import "@testing-library/jest-dom/vitest";

// jsdom has no object-URL support
if (!URL.createObjectURL) {
  URL.createObjectURL = () => "blob:mock";
  URL.revokeObjectURL = () => undefined;
}
