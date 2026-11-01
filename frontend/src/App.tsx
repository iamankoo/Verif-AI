import { useEffect, useState } from "react";
import { IconChip, IconClose, IconFile, IconGithub, IconHome, IconInfo, IconLayers, IconMenu } from "./components/Icons";
import { AboutPage, HowItWorksPage, ModelPage, ReportsPage } from "./pages/InfoPages";
import { AnalysisPage } from "./pages/AnalysisPage";
import { Home } from "./pages/Home";

export const REPO_URL = "https://github.com/iamankoo/Verif-AI";

export type Route = { page: "home" } | { page: "analysis"; id: string } | { page: "how" | "model" | "reports" | "about" };

export const parseRoute = (hash: string): Route => {
  const m = /^#\/analysis\/([0-9a-f]{32})$/.exec(hash);
  if (m) return { page: "analysis", id: m[1] };
  const named: Record<string, Route["page"]> = { "#/how-it-works": "how", "#/model": "model", "#/reports": "reports", "#/about": "about" };
  const p = named[hash];
  return p ? ({ page: p } as Route) : { page: "home" };
};

const NAV = [
  { href: "#/", label: "Home", page: "home", Icon: IconHome },
  { href: "#/how-it-works", label: "How It Works", page: "how", Icon: IconLayers },
  { href: "#/model", label: "Model", page: "model", Icon: IconChip },
  { href: "#/reports", label: "Reports", page: "reports", Icon: IconFile },
  { href: "#/about", label: "About", page: "about", Icon: IconInfo },
] as const;

function Header({ route }: { route: Route }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [route]);
  const current = route.page === "analysis" ? "home" : route.page;
  return (
    <header className="sticky top-0 z-30 border-b border-paper-line bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
        <a href="#/" className="flex items-center gap-2.5" aria-label="Verif-AI home">
          <img src="/favicon.svg" alt="" className="h-9 w-9" />
          <span className="text-xl font-bold tracking-tight">Verif-AI</span>
        </a>
        <nav aria-label="Main" className="hidden items-center gap-1 md:flex">
          {NAV.map(({ href, label, page, Icon }) => (
            <a key={page} href={href} className="nav-link" aria-current={current === page ? "page" : undefined}>
              <Icon className="h-4 w-4" />
              {label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="btn-secondary hidden sm:inline-flex">
            <IconGithub className="h-4 w-4" /> GitHub
          </a>
          <button className="btn-secondary !px-3 md:hidden" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} aria-controls="mobile-nav" onClick={() => setOpen((o) => !o)}>
            {open ? <IconClose className="h-5 w-5" /> : <IconMenu className="h-5 w-5" />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Main (mobile)" className="border-t border-paper-line px-4 pb-3 pt-2 md:hidden">
          <ul className="grid gap-1">
            {NAV.map(({ href, label, page, Icon }) => (
              <li key={page}>
                <a href={href} className="nav-link w-full" aria-current={current === page ? "page" : undefined}>
                  <Icon className="h-4 w-4" />
                  {label}
                </a>
              </li>
            ))}
            <li>
              <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="nav-link w-full">
                <IconGithub className="h-4 w-4" /> GitHub
              </a>
            </li>
          </ul>
        </nav>
      )}
    </header>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const on = () => {
      setRoute(parseRoute(window.location.hash));
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const open = (id: string) => {
    window.location.hash = `#/analysis/${id}`;
  };
  const home = () => {
    window.location.hash = "#/";
  };
  return (
    <>
      <Header route={route} />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {route.page === "analysis" ? <AnalysisPage id={route.id} onHome={home} /> : null}
        {route.page === "home" ? <Home onOpen={open} /> : null}
        {route.page === "how" ? <HowItWorksPage /> : null}
        {route.page === "model" ? <ModelPage /> : null}
        {route.page === "reports" ? <ReportsPage onOpen={open} /> : null}
        {route.page === "about" ? <AboutPage /> : null}
      </main>
      <footer className="mx-auto max-w-7xl px-4 pb-8 pt-2 text-xs text-ink-mute sm:px-6">
        Results are model-based evidence, not proof of manipulation or authenticity. Scores are uncalibrated. Verif-AI analyses faces only.
      </footer>
    </>
  );
}
