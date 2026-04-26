import React from "react";
import { DashboardView } from "../components/DashboardView.jsx";
import { collectDashboardSnapshot } from "../lib/dashboard.js";

export default function HomePage() {
  const snapshot = collectDashboardSnapshot();
  return <DashboardView snapshot={snapshot} />;
}
