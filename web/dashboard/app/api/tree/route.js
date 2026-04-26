import { listDirectoryEntries } from "../../../lib/dashboard.js";
import { resolveBrowserEntry } from "../../../lib/dashboard.js";

function normalizeSearchParam(requestUrl) {
  const searchParams = new URL(requestUrl).searchParams;
  return searchParams.get("path") || ".";
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export function GET(request) {
  try {
    const path = normalizeSearchParam(request.url);
    const entry = resolveBrowserEntry(process.env.DASHBOARD_PROJECT_ROOT, path);
    if (entry.type !== "directory") {
      return jsonResponse({ error: "tree_error", message: "Directory path required." }, 400);
    }
    return jsonResponse({ path, type: "directory", entries: listDirectoryEntries(process.env.DASHBOARD_PROJECT_ROOT, path) });
  } catch (error) {
    return jsonResponse({ error: "tree_error", message: "Requested path is invalid or unavailable." }, 400);
  }
}
