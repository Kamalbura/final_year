import React from "react";
import { BrowserView } from "../../components/BrowserView.jsx";
import { resolveBrowserEntry } from "../../lib/dashboard.js";

function normalizeSearchParam(searchParams) {
  const value = searchParams?.path;
  if (Array.isArray(value)) {
    return value[0] || ".";
  }
  return typeof value === "string" && value.trim() ? value : ".";
}

export default function BrowsePage({ searchParams }) {
  const projectRoot = process.env.DASHBOARD_PROJECT_ROOT;
  const relativePath = normalizeSearchParam(searchParams);

  try {
    const entry = resolveBrowserEntry(projectRoot, relativePath);
    return <BrowserView entry={entry} relativePath={relativePath} />;
  } catch (error) {
    return (
      <main className="shell">
        <section className="panel">
          <div className="panel-head">
            <h2>Browse {relativePath}</h2>
            <a href="/">Back to dashboard</a>
          </div>
          <p className="subtitle">Unable to load the requested path.</p>
          <pre className="code">Requested path is invalid or unavailable.</pre>
        </section>
      </main>
    );
  }
}
