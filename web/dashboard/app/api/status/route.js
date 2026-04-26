import { collectDashboardSnapshot } from "../../../lib/dashboard.js";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export function GET() {
  try {
    return jsonResponse(collectDashboardSnapshot());
  } catch {
    return jsonResponse({ error: "status_error", message: "Dashboard status is temporarily unavailable." }, 500);
  }
}
