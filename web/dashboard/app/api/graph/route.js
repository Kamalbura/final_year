import { collectDashboardSnapshot } from "../../../lib/dashboard.js";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export function GET() {
  try {
    const snapshot = collectDashboardSnapshot();
    return jsonResponse({
      generatedAt: snapshot.generatedAt,
      nodes: snapshot.lineage.map((node) => ({
        key: node.key,
        title: node.title,
        summary: node.summary,
        state: node.state,
        path: node.target.path,
        href: node.target.href,
      })),
      edges: snapshot.lineage.slice(1).map((node, index) => ({
        from: snapshot.lineage[index].key,
        to: node.key,
      })),
    });
  } catch {
    return jsonResponse({ error: "graph_error", message: "Dashboard graph is temporarily unavailable." }, 500);
  }
}
