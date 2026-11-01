import { describe, expect, it } from "vitest";
import { apiBase } from "./api";

describe("apiBase", () => {
  it("is same-origin /api when nothing is configured (dev proxy, single container)", () => {
    expect(apiBase({})).toBe("/api");
    expect(apiBase({ VITE_API_BASE_URL: "" })).toBe("/api");
  });
  it("prefixes the configured backend origin and tolerates trailing slashes and spaces", () => {
    expect(apiBase({ VITE_API_BASE_URL: "https://verifai-api.onrender.com" })).toBe("https://verifai-api.onrender.com/api");
    expect(apiBase({ VITE_API_BASE_URL: " https://verifai-api.onrender.com/// " })).toBe("https://verifai-api.onrender.com/api");
  });
  it("still honours the legacy VITE_API_BASE name, with the new name taking precedence", () => {
    expect(apiBase({ VITE_API_BASE: "https://old.example.com" })).toBe("https://old.example.com/api");
    expect(apiBase({ VITE_API_BASE_URL: "https://new.example.com", VITE_API_BASE: "https://old.example.com" })).toBe("https://new.example.com/api");
  });
});
