import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App, { parseRoute } from "./App";

const ID = "a".repeat(32);

describe("parseRoute", () => {
  it("maps hashes to pages and falls back to home", () => {
    expect(parseRoute("")).toEqual({ page: "home" });
    expect(parseRoute("#/how-it-works")).toEqual({ page: "how" });
    expect(parseRoute("#/model")).toEqual({ page: "model" });
    expect(parseRoute("#/reports")).toEqual({ page: "reports" });
    expect(parseRoute("#/about")).toEqual({ page: "about" });
    expect(parseRoute(`#/analysis/${ID}`)).toEqual({ page: "analysis", id: ID });
    expect(parseRoute("#/analysis/not-an-id")).toEqual({ page: "home" });
    expect(parseRoute("#/unknown")).toEqual({ page: "home" });
  });
});

describe("App shell", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("{}", { status: 503 })));
    window.location.hash = "";
  });
  afterEach(() => vi.unstubAllGlobals());

  it("has a main navigation with the five sections and marks the current page", () => {
    render(<App />);
    const nav = screen.getByRole("navigation", { name: "Main" });
    for (const n of ["Home", "How It Works", "Model", "Reports", "About"]) expect(nav).toHaveTextContent(n);
    expect(screen.getAllByRole("link", { name: "Home" })[0]).toHaveAttribute("aria-current", "page");
  });

  it("opens and closes the mobile menu", async () => {
    render(<App />);
    expect(screen.queryByRole("navigation", { name: /mobile/i })).toBeNull();
    await userEvent.click(screen.getByRole("button", { name: /open menu/i }));
    expect(screen.getByRole("navigation", { name: /mobile/i })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /close menu/i }));
    expect(screen.queryByRole("navigation", { name: /mobile/i })).toBeNull();
  });

  it("renders the about page for #/about without any invented statistics", () => {
    window.location.hash = "#/about";
    render(<App />);
    expect(screen.getByRole("heading", { name: /about verif-ai/i })).toBeInTheDocument();
    expect(screen.getByText(/not a universal AI-image or video detector/i)).toBeInTheDocument();
  });
});
