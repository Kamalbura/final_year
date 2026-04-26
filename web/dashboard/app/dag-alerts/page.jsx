import React from "react";
import { GET as getDagAlerts } from "../api/dag-alerts/route.js";

async function fetchDagAlerts() {
  const response = await getDagAlerts();

  const payload = await response.json();
  if (!response.ok) {
    return {
      ok: false,
      payload,
    };
  }

  return {
    ok: true,
    payload,
  };
}

function renderStat(label, value) {
  return (
    <div className="stat" key={label}>
      <span>{String(value)}</span>
      <small>{label}</small>
    </div>
  );
}

export const dynamic = "force-dynamic";

export default async function DagAlertsPage() {
  const result = await fetchDagAlerts();

  if (!result.ok) {
    return (
      <main className="shell">
        <section className="panel">
          <div className="panel-head">
            <h2>DAG Alerts</h2>
            <a href="/">Back to dashboard</a>
          </div>
          <p className="subtitle">DAG alerts are temporarily unavailable.</p>
          <pre className="code">{result.payload?.details ?? result.payload?.message ?? "unknown_error"}</pre>
        </section>
      </main>
    );
  }

  const { payload } = result;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Airflow monitoring</p>
          <h1>DAG Alerts</h1>
          <p className="subtitle">Live run failures and recent DAG state updates from Airflow.</p>
        </div>
        <div className="badge badge-warning">AUTO REFRESH: 30s</div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Totals</h2>
          <span>{payload.generatedAt}</span>
        </div>
        <div className="stats">
          {renderStat("Total DAGs", payload.totals.dags)}
          {renderStat("City DAGs", payload.totals.cityDags)}
          {renderStat("Paused DAGs", payload.totals.pausedDags)}
          {renderStat("Failed Runs", payload.totals.failedRuns)}
          {renderStat("Running Runs", payload.totals.runningRuns)}
          {renderStat("Queued Runs", payload.totals.queuedRuns)}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Recent Alert Messages</h2>
          <a href="/api/dag-alerts" target="_blank" rel="noreferrer">
            Open JSON
          </a>
        </div>
        <div className="chips">
          {payload.messages.length === 0 ? (
            <span className="chip chip-ready">
              <strong>No alerts</strong>
              <span>No recent run failures or queue/running updates.</span>
            </span>
          ) : (
            payload.messages.map((message) => (
              <span key={`${message.dagId}-${message.runId}-${message.timestamp}`} className={message.severity === "critical" ? "chip chip-missing" : "chip chip-ready"}>
                <strong>{message.state.toUpperCase()}</strong>
                <span>{message.text}</span>
              </span>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Failed Runs</h2>
          <a href="http://100.111.13.58:8080/home" target="_blank" rel="noreferrer">
            Open Airflow
          </a>
        </div>
        <table className="listing">
          <thead>
            <tr>
              <th>DAG</th>
              <th>Run ID</th>
              <th>State</th>
              <th>Logical Date</th>
              <th>Ended</th>
            </tr>
          </thead>
          <tbody>
            {payload.failedRuns.length === 0 ? (
              <tr>
                <td colSpan={5}>No failed runs in the latest sample window.</td>
              </tr>
            ) : (
              payload.failedRuns.map((run) => (
                <tr key={`${run.dagId}-${run.runId}`}>
                  <td>{run.dagId}</td>
                  <td>{run.runId}</td>
                  <td>{run.state}</td>
                  <td>{run.logicalDate ?? "-"}</td>
                  <td>{run.endDate ?? "-"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </section>

      <section className="footer">
        <a href="/">Back to dashboard</a>
        <a href="/api-docs">Open API docs</a>
      </section>

      <meta httpEquiv="refresh" content="30" />
    </main>
  );
}
