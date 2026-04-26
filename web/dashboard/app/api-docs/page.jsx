import React from "react";

const endpoints = [
  {
    method: "GET",
    path: "/api/status",
    purpose: "Repository health snapshot",
    response: "Overall status, lineage graph, quick links, and file/dir counts.",
  },
  {
    method: "GET",
    path: "/api/graph",
    purpose: "Lineage graph JSON",
    response: "Node list and edges for the repo architecture graph.",
  },
  {
    method: "GET",
    path: "/api/tree?path=<relative-path>",
    purpose: "Directory listing",
    response: "Safe directory entries for a resolved repo path.",
  },
  {
    method: "GET",
    path: "/api/file?path=<relative-path>",
    purpose: "File preview",
    response: "Metadata and a guarded preview for supported text files only.",
  },
  {
    method: "GET",
    path: "/api/dag-alerts",
    purpose: "Airflow DAG alerts and updates",
    response: "Recent run failures, queue/running updates, and DAG summary counters.",
  },
];

const isolationNotes = [
  "The Airflow scheduler now loads one DAG per city from deployment/pi_airflow/dags/aq_city_factory.py.",
  "A failure in one city run does not block retries for other cities.",
  "Each DAG writes to its own watermark and archive path, so reruns stay city-scoped.",
];

const cityCoverage = [
  "India: Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Surat, Kanpur, Nagpur, Bhopal, Visakhapatnam",
  "US: New York City, Los Angeles, Chicago, Houston, San Francisco",
  "UK: London, Manchester, Birmingham",
  "Other metros: Toronto, Vancouver, Paris, Berlin, Amsterdam, Dubai, Singapore, Tokyo, Seoul, Sydney, Melbourne, Lagos, Cape Town",
];

export default function ApiDocsPage() {
  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Dashboard API reference</p>
          <h1>API docs and AQI baseline</h1>
          <p className="subtitle">Read-only endpoints for repo status, graph view, directory browsing, and guarded file previews, plus the ingestion policy for city-isolated Airflow runs.</p>
        </div>
        <div className="badge badge-ready">READ ONLY</div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Endpoints</h2>
          <span>Safe JSON and preview routes</span>
        </div>
        <table className="listing">
          <thead>
            <tr>
              <th>Method</th>
              <th>Path</th>
              <th>Purpose</th>
              <th>Returns</th>
            </tr>
          </thead>
          <tbody>
            {endpoints.map((endpoint) => (
              <tr key={endpoint.path}>
                <td>{endpoint.method}</td>
                <td>{endpoint.path}</td>
                <td>{endpoint.purpose}</td>
                <td>{endpoint.response}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="grid">
        <article className="card">
          <div className="card-head">
            <div>
              <p className="card-kicker">Failure isolation</p>
              <h3>One city, one DAG</h3>
            </div>
          </div>
          <div className="chips">
            {isolationNotes.map((note) => (
              <span key={note} className="chip chip-ready">
                <strong>Note</strong>
                <span>{note}</span>
              </span>
            ))}
          </div>
        </article>

        <article className="card">
          <div className="card-head">
            <div>
              <p className="card-kicker">AQI baseline</p>
              <h3>Standardized, not universal</h3>
            </div>
          </div>
          <p className="subtitle-inline">The repo uses <code>us_aqi</code> as the standardized output field because Open-Meteo exposes that scale consistently. It is useful for cross-city comparison, but local health alerts still need city or country context.</p>
          <pre className="code">{`Interpretation rule:
- compare cities on the same scale
- keep the city's country/region visible
- avoid a single hard-coded global health threshold`}</pre>
        </article>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>City coverage</h2>
          <span>Curated major metros across regions</span>
        </div>
        <div className="chips">
          {cityCoverage.map((coverage) => (
            <span key={coverage} className="chip chip-ready">
              <strong>Catalog</strong>
              <span>{coverage}</span>
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Example response</h2>
          <span>/api/status</span>
        </div>
        <pre className="code">{`{
  "overallState": "ready",
  "lineage": [
    { "key": "frontend", "title": "Frontend", "state": "ready" },
    { "key": "airflow", "title": "Airflow", "state": "ready" }
  ],
  "quickLinks": [
    { "label": "City DAG factory", "path": "deployment/pi_airflow/dags/aq_city_factory.py" },
    { "label": "API docs", "path": "/api-docs" }
  ]
}`}</pre>
      </section>
    </main>
  );
}
