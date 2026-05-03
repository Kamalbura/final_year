"use client";
import React, { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const TARGET_CITIES = [
  { slug: "delhi", name: "Delhi NCR" },
  { slug: "hyderabad", name: "Hyderabad" },
  { slug: "bengaluru", name: "Bengaluru" },
];

export default function PredictionDashboard() {
  const [activeCity, setActiveCity] = useState("hyderabad");
  const [activeModel, setActiveModel] = useState("xgboost");
  const [forecastData, setForecastData] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/predictions/${activeCity}?model=${activeModel}`);
        if (!response.ok) throw new Error("Failed to fetch");
        const result = await response.json();
        setForecastData(result);
        if (result.available_models) setAvailableModels(result.available_models);
        setError(null);
      } catch (err) {
        setError(err.message);
        setForecastData(null);
      } finally {
        setLoading(false);
      }
    };
    fetchForecast();
  }, [activeCity, activeModel]);

  // Group models by category
  const groupedModels = availableModels.reduce((acc, m) => {
    if (!acc[m.category]) acc[m.category] = [];
    acc[m.category].push(m);
    return acc;
  }, {});

  const chartData = forecastData?.timeline?.map((item, i) => {
    const d = new Date(item.timestamp);
    return {
      time: d.toLocaleString('en-US', { weekday: 'short', hour: '2-digit' }),
      actual: item.actual_aqi || item.predicted_pm2_5,
      predicted: item.hour > 0 ? item.predicted_pm2_5 : null,
    };
  }) || [];

  return (
    <main className="predictions-page">
      <header className="page-header">
        <h1>PM₂.₅ Forecasts</h1>
        <p className="subtitle">24-hour predictions using ensemble & deep learning models</p>
      </header>

      {/* Selectors Row */}
      <div className="selectors-row">
        {/* City Tabs */}
        <div className="city-tabs">
          {TARGET_CITIES.map((city) => (
            <button key={city.slug} className={`tab-btn ${activeCity === city.slug ? "active" : ""}`} onClick={() => setActiveCity(city.slug)}>
              {city.name}
            </button>
          ))}
        </div>

        {/* Model Selector */}
        <div className="model-selector">
          <label>Model:</label>
          <select value={activeModel} onChange={(e) => setActiveModel(e.target.value)} className="model-dropdown">
            {Object.entries(groupedModels).map(([category, models]) => (
              <optgroup key={category} label={`── ${category} ──`}>
                {models.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.name} ({m.family})
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="loading">Loading {forecastData?.model?.name || activeModel} predictions...</div>
      ) : error ? (
        <div className="error">Error: {error}</div>
      ) : forecastData?.error ? (
        <div className="info-message">{forecastData.message || "Awaiting prediction"}</div>
      ) : forecastData ? (
        <div className="dashboard-content">
          {/* Top Stats */}
          <section className="top-stats">
            <div className="stat-card">
              <h3>Current PM₂.₅</h3>
              <div className="value">{forecastData.current_pm2_5?.toFixed(1) || "—"} <span>µg/m³</span></div>
            </div>
            <div className="stat-card highlight">
              <h3>24h Forecast Avg</h3>
              <div className="value">{forecastData.forecast_summary?.next_24h?.toFixed(1) || "—"} <span>µg/m³</span></div>
              <div className="sub">{forecastData.forecast_summary?.trend_label}</div>
            </div>
            <div className="stat-card model-info">
              <h3>Active Model</h3>
              <div className="value" style={{fontSize: '1.2rem'}}>{forecastData.model?.name}</div>
              <div className="sub">{forecastData.model?.family} · Tier {forecastData.model?.tier}</div>
            </div>
            <div className="stat-card">
              <h3>Generated</h3>
              <div className="value" style={{fontSize: '1rem'}}>
                {forecastData.generated_at ? new Date(forecastData.generated_at).toLocaleTimeString() : "—"}
              </div>
              <div className="sub">Updates hourly</div>
            </div>
          </section>

          {/* Chart */}
          <section className="chart-container">
            <h2>24-Hour PM₂.₅ Forecast</h2>
            <div style={{ width: '100%', height: 380 }}>
              <ResponsiveContainer>
                <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis dataKey="time" tick={{ fontSize: 12, fill: '#9ca3af' }} />
                  <YAxis label={{ value: 'PM₂.₅ µg/m³', angle: -90, position: 'insideLeft', fill: '#9ca3af', style: { fontSize: 12 } }} tick={{ fill: '#9ca3af' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: 8, color: '#fff' }} />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#10b981" strokeWidth={3} name="PM₂.₅" dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="predicted" stroke="#3b82f6" strokeWidth={3} strokeDasharray="5 5" name="Predicted" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Insights */}
          <section className="ai-advisory">
            <h2>Insights</h2>
            <p>{forecastData.insights}</p>
          </section>
        </div>
      ) : null}

      <style jsx>{`
        .predictions-page { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .page-header h1 { font-size: 2.2rem; margin-bottom: 5px; }
        .subtitle { color: var(--muted); margin-bottom: 20px; }
        .selectors-row { display: flex; gap: 20px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }
        .city-tabs { display: flex; gap: 8px; }
        .tab-btn { background: transparent; border: none; color: var(--muted); font-size: 1rem; padding: 8px 18px; cursor: pointer; border-radius: 8px; transition: all 0.2s; }
        .tab-btn:hover { background: rgba(255,255,255,0.05); color: var(--text); }
        .tab-btn.active { background: var(--accent); color: var(--bg); font-weight: 600; }
        .model-selector { display: flex; align-items: center; gap: 8px; }
        .model-selector label { color: var(--muted); font-size: 0.9rem; }
        .model-dropdown { background: var(--panel); color: var(--text); border: 1px solid var(--panel-border); border-radius: 8px; padding: 8px 12px; font-size: 0.9rem; min-width: 250px; cursor: pointer; }
        .model-dropdown optgroup { font-weight: 600; color: var(--accent); }
        .model-dropdown option { color: var(--text); background: var(--bg); padding: 4px; }
        .top-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 12px; padding: 16px; text-align: center; }
        .stat-card h3 { margin: 0 0 8px 0; color: var(--muted); font-size: 0.85rem; text-transform: uppercase; }
        .stat-card .value { font-size: 1.8rem; font-weight: bold; color: var(--text); }
        .stat-card .value span { font-size: 0.8rem; color: var(--muted); }
        .stat-card .sub { color: var(--muted); font-size: 0.8rem; margin-top: 4px; }
        .stat-card.highlight { border-color: var(--accent); background: rgba(59,130,246,0.08); }
        .stat-card.highlight .value { color: var(--accent); }
        .chart-container { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .chart-container h2 { margin: 0 0 15px 0; }
        .ai-advisory { background: var(--panel); border: 1px solid var(--panel-border); border-left: 4px solid #10b981; border-radius: 12px; padding: 20px; }
        .ai-advisory h2 { margin: 0 0 10px 0; }
        .ai-advisory p { color: var(--text); line-height: 1.6; margin: 0; }
        .loading, .error, .info-message { text-align: center; padding: 40px; color: var(--muted); font-size: 1.1rem; background: var(--panel); border-radius: 12px; }
        @media (max-width: 768px) { .top-stats { grid-template-columns: 1fr; } .selectors-row { flex-direction: column; align-items: flex-start; } }
      `}</style>
    </main>
  );
}
