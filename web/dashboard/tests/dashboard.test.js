import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { renderToStaticMarkup } from "react-dom/server";

import { DashboardView } from "../components/DashboardView.jsx";
import { BrowserView } from "../components/BrowserView.jsx";
import { collectDashboardSnapshot, listDirectoryEntries, readFileDescriptor, resolveBrowserEntry } from "../lib/dashboard.js";
import HomePage from "../app/page.jsx";
import BrowsePage from "../app/browse/page.jsx";
import ApiDocsPage from "../app/api-docs/page.jsx";
import { GET as getStatus } from "../app/api/status/route.js";
import { GET as getGraph } from "../app/api/graph/route.js";
import { GET as getTree } from "../app/api/tree/route.js";
import { GET as getFile } from "../app/api/file/route.js";

function seedRepo(root) {
  const files = {
    "web/dashboard/package.json": '{"name":"dashboard"}\n',
    "web/dashboard/app/page.jsx": "export default function Page() {}\n",
    "web/dashboard/app/browse/page.jsx": "export default function Page() {}\n",
    "web/dashboard/app/api-docs/page.jsx": "export default function Page() {}\n",
    "deployment/pi_airflow/docker-compose.yml": "services: {}\n",
    "deployment/pi_airflow/dags/aq_city_factory.py": "print('factory')\n",
    "deployment/pi_airflow/sql/india_aq_schema.sql": "create schema aq;\n",
    "scripts/bootstrap_india_aq_db.py": "print('bootstrap')\n",
    "requirements.txt": "requests\n",
    "src/ingestion/india_aq.py": "print('ingestion')\n",
    "tests/test_india_aq.py": "print('tests')\n",
    "web/dashboard/tests/dashboard.test.js": "test('ok',()=>{})\n",
    "data/india_aq_1y/sample.csv": "city,timestamp\nDelhi,2026-04-25T00:00:00Z\n",
    "data/hyderabad_station_aq_1y/sample.csv": "city,timestamp\nHyderabad,2026-04-25T00:00:00Z\n",
    "README.md": "# demo\n",
    ".secret.txt": "hidden\n",
    "notes.txt": "hello world\n",
  };

  for (const [relativePath, content] of Object.entries(files)) {
    const filePath = path.join(root, relativePath);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, content);
  }
}

async function withDashboardRoot(root, fn) {
  const previousRoot = process.env.DASHBOARD_PROJECT_ROOT;
  process.env.DASHBOARD_PROJECT_ROOT = root;
  try {
    return await fn();
  } finally {
    if (previousRoot === undefined) {
      delete process.env.DASHBOARD_PROJECT_ROOT;
    } else {
      process.env.DASHBOARD_PROJECT_ROOT = previousRoot;
    }
  }
}

test("collector builds a stable snapshot", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dashboard-root-"));
  seedRepo(root);

  const snapshot = collectDashboardSnapshot(root);

  assert.equal(snapshot.lineage.map((node) => node.key).join(","), "frontend,backend,airflow,archive");
  assert.equal(snapshot.cards.length, 5);
  assert.ok(snapshot.quickLinks.some((target) => target.path === "/api-docs"));
  assert.equal(snapshot.cards[0].state, "ready");
  assert.equal(snapshot.stats.csvFiles, 2);
});

test("collector rejects traversal and hides dotfiles", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dashboard-root-"));
  seedRepo(root);

  assert.throws(() => listDirectoryEntries(root, "../windows"));
  const entries = listDirectoryEntries(root, ".");
  assert.ok(entries.every((entry) => !entry.name.startsWith(".")));
});

test("collector reads safe previews only", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dashboard-root-"));
  seedRepo(root);

  const file = readFileDescriptor(root, "deployment/pi_airflow/docker-compose.yml");
  assert.equal(file.kind, "file");
  assert.match(file.preview.join("\n"), /services/);

  const unsupported = readFileDescriptor(root, ".secret.txt");
  assert.match(unsupported.preview[0], /Preview disabled|File too large/);
});

test("route handlers return the expected json payloads", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dashboard-root-"));
  seedRepo(root);

  await withDashboardRoot(root, async () => {
    const statusResponse = await getStatus();
    const statusPayload = await statusResponse.json();
    assert.equal(statusResponse.status, 200);
    assert.equal(statusPayload.overallState, "ready");

    const graphResponse = await getGraph();
    const graphPayload = await graphResponse.json();
    assert.equal(graphPayload.nodes[0].key, "frontend");

    const treeResponse = await getTree(new Request("http://localhost/api/tree?path=deployment/pi_airflow"));
    const treePayload = await treeResponse.json();
    assert.ok(treePayload.entries.some((entry) => entry.name === "docker-compose.yml"));

    const fileResponse = await getFile(new Request("http://localhost/api/file?path=deployment/pi_airflow/docker-compose.yml"));
    const filePayload = await fileResponse.json();
    assert.equal(filePayload.entry.name, "docker-compose.yml");
  });
});

test("react pages render the dashboard and browser views", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dashboard-root-"));
  seedRepo(root);

  return withDashboardRoot(root, () => {
    const homeMarkup = renderToStaticMarkup(HomePage());
    assert.match(homeMarkup, /React and Node view/);
    assert.match(homeMarkup, /Dashboard status API/);
    assert.match(homeMarkup, /Open API docs/);

    const docsMarkup = renderToStaticMarkup(ApiDocsPage());
    assert.match(docsMarkup, /API docs and AQI baseline/);
    assert.match(docsMarkup, /one city, one DAG/i);

    const browserMarkup = renderToStaticMarkup(BrowsePage({ searchParams: { path: "deployment/pi_airflow" } }));
    assert.match(browserMarkup, /docker-compose.yml/);
    assert.match(browserMarkup, /Back to dashboard/);

    const fileMarkup = renderToStaticMarkup(BrowsePage({ searchParams: { path: "deployment/pi_airflow/docker-compose.yml" } }));
    assert.match(fileMarkup, /docker-compose.yml/);
    assert.match(fileMarkup, /Up one level/);

    const dashboardMarkup = renderToStaticMarkup(
      DashboardView({
        snapshot: collectDashboardSnapshot(root),
      })
    );
    assert.match(dashboardMarkup, /Open repo browser/);
    assert.match(dashboardMarkup, /Open API docs/);
  });
});

test("resolveBrowserEntry returns file and directory variants", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "dashboard-root-"));
  seedRepo(root);

  const directory = resolveBrowserEntry(root, "deployment/pi_airflow");
  assert.equal(directory.type, "directory");
  const file = resolveBrowserEntry(root, "deployment/pi_airflow/docker-compose.yml");
  assert.equal(file.type, "file");
});
