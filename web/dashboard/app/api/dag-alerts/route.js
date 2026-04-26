function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function toIsoOrNull(value) {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date.toISOString();
}

function airflowHeaders() {
  const username = process.env.AIRFLOW_API_USERNAME;
  const password = process.env.AIRFLOW_API_PASSWORD;
  if (!username || !password) {
    return {};
  }
  const token = Buffer.from(`${username}:${password}`).toString("base64");
  return { Authorization: `Basic ${token}` };
}

async function fetchAirflowJson(pathname) {
  const base = process.env.AIRFLOW_API_BASE_URL ?? "http://100.111.13.58:8080/api/v1";
  const url = `${base.replace(/\/$/, "")}${pathname}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "application/json",
      ...airflowHeaders(),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Airflow API ${response.status}: ${text.slice(0, 300)}`);
  }

  return response.json();
}

function summarizeState(runs) {
  const counters = {
    queued: 0,
    running: 0,
    failed: 0,
    success: 0,
    other: 0,
  };

  for (const run of runs) {
    const state = run.state;
    if (state === "queued" || state === "running" || state === "failed" || state === "success") {
      counters[state] += 1;
    } else {
      counters.other += 1;
    }
  }

  return counters;
}

function buildMessages(runs) {
  const sorted = [...runs].sort((left, right) => {
    const leftDate = new Date(left.logical_date ?? left.execution_date ?? 0).getTime();
    const rightDate = new Date(right.logical_date ?? right.execution_date ?? 0).getTime();
    return rightDate - leftDate;
  });

  const messages = [];
  for (const run of sorted.slice(0, 25)) {
    const when = toIsoOrNull(run.logical_date ?? run.execution_date ?? run.start_date);
    if (run.state === "failed") {
      messages.push({
        severity: "critical",
        dagId: run.dag_id,
        runId: run.dag_run_id,
        state: run.state,
        timestamp: when,
        text: `${run.dag_id} failed at ${when ?? "unknown time"}`,
      });
      continue;
    }

    if (run.state === "running" || run.state === "queued") {
      messages.push({
        severity: "info",
        dagId: run.dag_id,
        runId: run.dag_run_id,
        state: run.state,
        timestamp: when,
        text: `${run.dag_id} is currently ${run.state}`,
      });
    }
  }

  return messages;
}

export async function GET() {
  try {
    const dagsPayload = await fetchAirflowJson("/dags?limit=500&only_active=false");
    const runsPayload = await fetchAirflowJson("/dags/~/dagRuns?limit=500&order_by=-logical_date");

    const dags = Array.isArray(dagsPayload.dags) ? dagsPayload.dags : [];
    const runs = Array.isArray(runsPayload.dag_runs) ? runsPayload.dag_runs : [];

    const counters = summarizeState(runs);
    const failedRuns = runs
      .filter((run) => run.state === "failed")
      .sort((left, right) => {
        const leftDate = new Date(left.logical_date ?? left.execution_date ?? 0).getTime();
        const rightDate = new Date(right.logical_date ?? right.execution_date ?? 0).getTime();
        return rightDate - leftDate;
      })
      .slice(0, 50)
      .map((run) => ({
        dagId: run.dag_id,
        runId: run.dag_run_id,
        state: run.state,
        startDate: toIsoOrNull(run.start_date),
        endDate: toIsoOrNull(run.end_date),
        logicalDate: toIsoOrNull(run.logical_date ?? run.execution_date),
      }));

    const cityDags = dags.filter((dag) => typeof dag.dag_id === "string" && dag.dag_id.startsWith("aq_"));

    return jsonResponse({
      generatedAt: new Date().toISOString(),
      source: "airflow_api",
      totals: {
        dags: dags.length,
        cityDags: cityDags.length,
        pausedDags: dags.filter((dag) => dag.is_paused).length,
        failedRuns: counters.failed,
        runningRuns: counters.running,
        queuedRuns: counters.queued,
        successRuns: counters.success,
      },
      failedRuns,
      messages: buildMessages(runs),
      recentRuns: runs.slice(0, 75).map((run) => ({
        dagId: run.dag_id,
        runId: run.dag_run_id,
        state: run.state,
        logicalDate: toIsoOrNull(run.logical_date ?? run.execution_date),
        startDate: toIsoOrNull(run.start_date),
        endDate: toIsoOrNull(run.end_date),
      })),
    });
  } catch (error) {
    return jsonResponse(
      {
        error: "dag_alerts_unavailable",
        message: "Unable to fetch DAG alerts from Airflow right now.",
        details: error instanceof Error ? error.message : "unknown_error",
      },
      502
    );
  }
}
