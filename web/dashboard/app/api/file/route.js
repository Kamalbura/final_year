import { readFileDescriptor } from "../../../lib/dashboard.js";

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export function GET(request) {
  try {
    const searchParams = new URL(request.url).searchParams;
    const path = searchParams.get("path") || ".";
    const entry = readFileDescriptor(process.env.DASHBOARD_PROJECT_ROOT, path);
    return jsonResponse({ path, type: "file", entry });
  } catch (error) {
    return jsonResponse({ error: "file_error", message: "Requested file is invalid or unavailable." }, 400);
  }
}
