import fs from "node:fs";
import path from "node:path";

const SAFE_PREVIEW_EXTENSIONS = new Set([
  ".csv",
  ".ini",
  ".json",
  ".js",
  ".jsx",
  ".md",
  ".mjs",
  ".py",
  ".sh",
  ".sql",
  ".txt",
  ".ts",
  ".tsx",
  ".yaml",
  ".yml",
]);

const STATUS_TARGETS = [
  {
    key: "frontend",
    title: "Frontend shell",
    relatives: ["web/dashboard", "web/dashboard/package.json", "web/dashboard/app/page.jsx", "web/dashboard/app/api-docs/page.jsx"],
  },
  {
    key: "backend",
    title: "Backend project core",
    relatives: ["src", "scripts", "requirements.txt"],
  },
  {
    key: "airflow",
    title: "Airflow deployment",
    relatives: [
      "deployment/pi_airflow",
      "deployment/pi_airflow/docker-compose.yml",
        "deployment/pi_airflow/dags/aq_city_factory.py",
    ],
  },
  {
    key: "storage",
    title: "Database and archive",
    relatives: [
      "deployment/pi_airflow/sql/india_aq_schema.sql",
      "scripts/bootstrap_india_aq_db.py",
      "data/india_aq_1y",
      "data/hyderabad_station_aq_1y",
    ],
  },
  {
    key: "tests",
    title: "Verification suite",
    relatives: ["tests", "tests/test_india_aq.py", "web/dashboard/tests/dashboard.test.js"],
  },
];

const LINEAGE_TARGETS = [
  {
    key: "frontend",
    title: "Frontend",
    summary: "React dashboard and repo browser",
    relatives: ["web/dashboard", "web/dashboard/app/page.jsx"],
  },
  {
    key: "backend",
    title: "Backend",
    summary: "Python tools, ingestion, and data utilities",
    relatives: ["src", "scripts", "requirements.txt"],
  },
  {
    key: "airflow",
    title: "Airflow",
    summary: "City-isolated scheduler and DAG factory",
    relatives: ["deployment/pi_airflow", "deployment/pi_airflow/dags/aq_city_factory.py"],
  },
  {
    key: "archive",
    title: "Database and archive",
    summary: "Schema, bootstrap, and data snapshots",
    relatives: ["deployment/pi_airflow/sql/india_aq_schema.sql", "scripts/bootstrap_india_aq_db.py", "data/india_aq_1y"],
  },
];

function defaultProjectRoot() {
  return path.resolve(process.env.DASHBOARD_PROJECT_ROOT ?? path.resolve(process.cwd(), "..", ".."));
}

function normalizeForCompare(value) {
  const normalized = path.resolve(value).replace(/\\/g, "/").replace(/\/+$/, "");
  return process.platform === "win32" ? normalized.toLowerCase() : normalized;
}

function isInsideRoot(rootPath, candidatePath) {
  const root = normalizeForCompare(rootPath);
  const candidate = normalizeForCompare(candidatePath);
  return candidate === root || candidate.startsWith(`${root}/`);
}

export function resolveInsideRoot(projectRoot, relativePath = ".") {
  const root = path.resolve(projectRoot ?? defaultProjectRoot());
  const candidate = path.resolve(root, relativePath || ".");
  if (!isInsideRoot(root, candidate)) {
    throw new Error(`Path escapes project root: ${relativePath}`);
  }
  return candidate;
}

function exists(projectRoot, relativePath) {
  try {
    return fs.existsSync(resolveInsideRoot(projectRoot, relativePath));
  } catch {
    return false;
  }
}

function hasSymlinkAncestor(projectRoot, relativePath) {
  const resolved = resolveInsideRoot(projectRoot, relativePath);
  if (!fs.existsSync(resolved)) {
    return false;
  }
  const root = path.resolve(projectRoot ?? defaultProjectRoot());
  const relativeParts = path.relative(root, resolved).split(path.sep).filter(Boolean);
  let current = root;
  for (const part of relativeParts) {
    current = path.join(current, part);
    if (fs.lstatSync(current).isSymbolicLink()) {
      return true;
    }
  }
  return false;
}

function resolveExistingInsideRoot(projectRoot, relativePath) {
  const resolved = resolveInsideRoot(projectRoot, relativePath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`Path not found: ${relativePath}`);
  }
  if (hasSymlinkAncestor(projectRoot, relativePath)) {
    throw new Error(`Path is a symbolic link: ${relativePath}`);
  }
  const rootReal = fs.realpathSync(path.resolve(projectRoot ?? defaultProjectRoot()));
  const real = fs.realpathSync(resolved);
  if (!isInsideRoot(rootReal, real)) {
    throw new Error(`Path escapes project root: ${relativePath}`);
  }
  return real;
}

function formatBytes(bytesCount) {
  if (bytesCount < 1024) {
    return `${bytesCount} B`;
  }
  if (bytesCount < 1024 * 1024) {
    return `${(bytesCount / 1024).toFixed(1)} KB`;
  }
  return `${(bytesCount / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTimestamp(date) {
  return date.toISOString();
}

function walkFiles(rootPath) {
  const collected = [];
  const stack = [rootPath];
  while (stack.length > 0) {
    const current = stack.pop();
    if (!fs.existsSync(current)) {
      continue;
    }
    const stat = fs.lstatSync(current);
    if (stat.isSymbolicLink()) {
      continue;
    }
    if (stat.isFile()) {
      collected.push(current);
      continue;
    }
    if (!stat.isDirectory()) {
      continue;
    }
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      if (entry.name.startsWith(".")) {
        continue;
      }
      stack.push(path.join(current, entry.name));
    }
  }
  return collected;
}

function buildTarget(projectRoot, label, relativePath, kind) {
  const present = kind === "external" ? true : exists(projectRoot, relativePath);
  return {
    label,
    path: relativePath,
    kind,
    exists: present,
    href: kind === "external" ? relativePath : `/browse?path=${encodeURIComponent(relativePath)}`,
  };
}

function buildGroupStatus(projectRoot, { key, title, relatives }) {
  const targets = relatives.map((relativePath) => buildTarget(projectRoot, path.basename(relativePath) || relativePath, relativePath, path.extname(relativePath) ? "file" : "directory"));
  const presentCount = targets.filter((target) => target.exists).length;
  const state = presentCount === targets.length ? "ready" : presentCount > 0 ? "warning" : "missing";
  return {
    key,
    title,
    state,
    summary: `${presentCount}/${targets.length} paths present`,
    target: targets[0],
    evidence: targets,
  };
}

function buildLineageNode(projectRoot, { key, title, summary, relatives }) {
  const target = buildTarget(projectRoot, title, relatives[0], path.extname(relatives[0]) ? "file" : "directory");
  const ready = relatives.every((relativePath) => exists(projectRoot, relativePath));
  return {
    key,
    title,
    summary,
    target,
    state: ready ? "ready" : "warning",
  };
}

function countFilesMatching(rootPath, predicate) {
  return walkFiles(rootPath).filter(predicate).length;
}

function latestFileUpdate(rootPath) {
  const files = walkFiles(rootPath);
  if (files.length === 0) {
    return null;
  }
  let latest = 0;
  for (const filePath of files) {
    const modifiedAt = fs.statSync(filePath).mtimeMs;
    if (modifiedAt > latest) {
      latest = modifiedAt;
    }
  }
  return formatTimestamp(new Date(latest));
}

function countDataDirectories(projectRoot) {
  const dataRoot = resolveInsideRoot(projectRoot, "data");
  if (!fs.existsSync(dataRoot) || !fs.lstatSync(dataRoot).isDirectory()) {
    return 0;
  }
  return fs.readdirSync(dataRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory() && !entry.name.startsWith(".")).length;
}

export function collectDashboardSnapshot(projectRoot = defaultProjectRoot()) {
  const resolvedRoot = path.resolve(projectRoot);
  const cards = STATUS_TARGETS.map((target) => buildGroupStatus(resolvedRoot, target));
  const lineage = LINEAGE_TARGETS.map((target) => buildLineageNode(resolvedRoot, target));
  const states = new Set(cards.map((card) => card.state));
  const overallState = states.size === 1 && states.has("ready") ? "ready" : "warning";

  return {
    generatedAt: formatTimestamp(new Date()),
    projectRootLabel: path.basename(resolvedRoot) || ".",
    overallState,
    cards,
    lineage,
    quickLinks: [
      buildTarget(resolvedRoot, "Dashboard app", "web/dashboard", "directory"),
      buildTarget(resolvedRoot, "City DAG factory", "deployment/pi_airflow/dags/aq_city_factory.py", "file"),
      buildTarget(resolvedRoot, "API docs", "/api-docs", "external"),
      buildTarget(resolvedRoot, "DAG alerts", "/dag-alerts", "external"),
      buildTarget(resolvedRoot, "Schema", "deployment/pi_airflow/sql/india_aq_schema.sql", "file"),
      buildTarget(resolvedRoot, "Bootstrap script", "scripts/bootstrap_india_aq_db.py", "file"),
      buildTarget(resolvedRoot, "Tests", "tests", "directory"),
      buildTarget(resolvedRoot, "Dashboard status API", "/api/status", "external"),
      buildTarget(resolvedRoot, "DAG alerts API", "/api/dag-alerts", "external"),
    ],
    stats: {
      pythonFiles: countFilesMatching(resolvedRoot, (filePath) => filePath.endsWith(".py")),
      csvFiles: countFilesMatching(resolvedRoot, (filePath) => filePath.endsWith(".csv")),
      notebooks: countFilesMatching(resolvedRoot, (filePath) => filePath.endsWith(".ipynb")),
      dataDirs: countDataDirectories(resolvedRoot),
      airflowDags: countFilesMatching(resolveInsideRoot(resolvedRoot, "deployment/pi_airflow"), (filePath) => filePath.endsWith(".py")),
      latestRepoUpdate: latestFileUpdate(resolvedRoot),
    },
  };
}

export function listDirectoryEntries(projectRoot, relativePath = ".") {
  const resolved = resolveExistingInsideRoot(projectRoot, relativePath);
  const stat = fs.statSync(resolved);
  if (!stat.isDirectory()) {
    return [readFileDescriptor(projectRoot, relativePath)];
  }
  return fs
    .readdirSync(resolved, { withFileTypes: true })
    .filter((entry) => !entry.name.startsWith(".") && !entry.isSymbolicLink())
    .map((entry) => {
      const entryPath = path.join(resolved, entry.name);
      const entryRelativePath = path.relative(path.resolve(projectRoot), entryPath).replace(/\\/g, "/");
      const entryStat = fs.statSync(entryPath);
      return {
        name: entry.name,
        kind: entry.isDirectory() ? "directory" : "file",
        size: formatBytes(entryStat.size),
        modified: formatTimestamp(entryStat.mtime),
        relativePath: entryRelativePath,
        href: `/browse?path=${encodeURIComponent(entryRelativePath)}`,
      };
    })
    .sort((left, right) => {
      if (left.kind !== right.kind) {
        return left.kind === "directory" ? -1 : 1;
      }
      return left.name.localeCompare(right.name);
    });
}

export function readFileDescriptor(projectRoot, relativePath) {
  const resolved = resolveExistingInsideRoot(projectRoot, relativePath);
  const stat = fs.statSync(resolved);
  if (!stat.isFile()) {
    throw new Error(`Not a file: ${relativePath}`);
  }
  const extension = path.extname(resolved).toLowerCase();
  const safePreview = SAFE_PREVIEW_EXTENSIONS.has(extension) && !path.basename(resolved).startsWith(".");
  let preview = ["Preview disabled for this file type."];
  if (safePreview) {
    if (stat.size <= 64 * 1024) {
      preview = fs.readFileSync(resolved, "utf8").split(/\r?\n/).slice(0, 200);
    } else {
      preview = ["File too large to preview safely."];
    }
  }
  const parentRelativePath = path.relative(path.resolve(projectRoot), path.dirname(resolved)).replace(/\\/g, "/");
  return {
    name: path.basename(resolved),
    kind: "file",
    size: formatBytes(stat.size),
    modified: formatTimestamp(stat.mtime),
    relativePath,
    parentPath: parentRelativePath === "." ? "" : parentRelativePath,
    preview,
    href: `/browse?path=${encodeURIComponent(relativePath)}`,
  };
}

export function resolveBrowserEntry(projectRoot, relativePath = ".") {
  const resolved = resolveExistingInsideRoot(projectRoot, relativePath);
  const stat = fs.statSync(resolved);
  if (stat.isFile()) {
    return { type: "file", data: readFileDescriptor(projectRoot, relativePath) };
  }
  return { type: "directory", data: listDirectoryEntries(projectRoot, relativePath) };
}
