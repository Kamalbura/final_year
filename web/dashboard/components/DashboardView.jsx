import React from "react";

function badgeClassName(state) {
  if (state === "ready") {
    return "badge badge-ready";
  }
  if (state === "missing") {
    return "badge badge-missing";
  }
  return "badge badge-warning";
}

function linkChip(target) {
  const badge = target.exists ? "chip chip-ready" : "chip chip-missing";
  return (
    <a key={`${target.path}-${target.label}`} className={badge} href={target.href} target={target.kind === "external" ? "_blank" : undefined} rel={target.kind === "external" ? "noreferrer" : undefined}>
      <strong>{target.label}</strong>
      <span>{target.path}</span>
    </a>
  );
}

export function DashboardView({ snapshot }) {
  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Operational status dashboard</p>
          <h1>Air Quality Project</h1>
          <p className="subtitle">React and Node view for deployment status, Airflow lineage, and repo navigation.</p>
        </div>
        <div className={badgeClassName(snapshot.overallState)}>{snapshot.overallState.toUpperCase()}</div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Lineage Flow</h2>
          <a href="/api/graph">View JSON</a>
        </div>
        <div className="lineage">
          {snapshot.lineage.map((node, index) => (
            <a key={node.key} className={`lineage-node node-${node.state}`} href={node.target.href}>
              <strong>{node.title}</strong>
              <span>{node.summary}</span>
              <small>{node.target.path}</small>
              {index < snapshot.lineage.length - 1 ? <i className="arrow">→</i> : null}
            </a>
          ))}
        </div>
      </section>

      <section className="grid">
        {snapshot.cards.map((card) => (
          <article key={card.key} className={`card card-${card.state}`}>
            <div className="card-head">
              <div>
                <p className="card-kicker">{card.key}</p>
                <h3>{card.title}</h3>
              </div>
              <span className={badgeClassName(card.state)}>{card.state}</span>
            </div>
            <p>{card.summary}</p>
            <div className="chips">{linkChip(card.target)}</div>
            <div className="chips evidence">{card.evidence.map(linkChip)}</div>
          </article>
        ))}
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Quick Links</h2>
          <a href="/api/status">Snapshot JSON</a>
        </div>
        <div className="chips">
          {snapshot.quickLinks.map((target) => linkChip(target))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Project Signals</h2>
          <span>Generated {snapshot.generatedAt}</span>
        </div>
        <div className="stats">
          {Object.entries(snapshot.stats).map(([key, value]) => (
            <div key={key} className="stat">
              <span>{String(value)}</span>
              <small>{key.replace(/([A-Z])/g, " $1").trim()}</small>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Current State</h2>
          <span>{snapshot.projectRootLabel}</span>
        </div>
        <div className="chips">
          <a className="chip chip-ready" href="/browse?path=.">
            <strong>Open repo browser</strong>
            <span>Browse files and folders</span>
          </a>
          <a className="chip chip-ready" href="/api-docs">
            <strong>Open API docs</strong>
            <span>Dashboard endpoints and AQI baseline</span>
          </a>
          <a className="chip chip-ready" href="/api/status" target="_blank" rel="noreferrer">
            <strong>Dashboard status API</strong>
            <span>/api/status</span>
          </a>
        </div>
      </section>
    </main>
  );
}
