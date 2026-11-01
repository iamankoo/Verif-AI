import { useEffect, useState } from "react";
import { AnalysisPage } from "./pages/AnalysisPage";
import { Home } from "./pages/Home";

const parse = (hash: string): { id: string } | null => {
  const m = /^#\/analysis\/([0-9a-f]{32})$/.exec(hash);
  return m ? { id: m[1] } : null;
};

function Brand() {
  return (
    <header className="border-b border-paper-line bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <a href="#/" className="flex items-center gap-3" aria-label="Verif-AI home">
          <img src="/favicon.svg" alt="" className="h-9 w-9" />
          <span>
            <span className="block font-serif text-xl leading-tight">Verif-AI</span>
            <span className="block text-xs text-ink-mute">See Beyond the Pixels. Verify What's Real.</span>
          </span>
        </a>
        <span className="hidden text-xs text-ink-mute sm:block">Synthetic media detection &amp; forensic analysis</span>
      </div>
    </header>
  );
}

export default function App() {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
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
      <Brand />
      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-6">{route ? <AnalysisPage id={route.id} onHome={home} /> : <Home onOpen={open} />}</main>
      <footer className="mx-auto max-w-6xl px-4 pb-8 text-xs text-ink-mute sm:px-6">
        Results are model-based evidence, not proof of manipulation or authenticity. Scores are uncalibrated.
      </footer>
    </>
  );
}
