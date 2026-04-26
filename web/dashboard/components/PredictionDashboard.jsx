/**
 * Component: Predictions Dashboard
 * Route: /predictions
 * 
 * Shows:
 * - Tabs for 4 target cities
 * - Real-time forecast vs actual chart
 * - Model info and confidence
 */

"use client";

import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart
} from "recharts";

const TARGET_CITIES = [
  { slug: "delhi", name: "Delhi NCR" },
  { slug: "mumbai", name: "Mumbai" },
  { slug: "hyderabad", name: "Hyderabad (UoH)" },
  { slug: "bengaluru", name: "Bengaluru (MG Road)" },
];

const AQI_COLORS = {
  good: "#10b981",
  moderate: "#f59e0b",
  unhealthy_sg: "#f97316",
  unhealthy: "#ef4444",
  very_unhealthy: "#9333ea",
  hazardous: "#7c2d12"
};

export default function PredictionDashboard() {
  const [activeCity, setActiveCity] = useState("delhi");
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/predictions/${activeCity}`);
        if (!response.ok) throw new Error("Failed to fetch forecast data");
        const result = await response.json();
        setForecastData(result);
        setError(null);
      } catch (err) {
        setError(err.message);
        setForecastData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchForecast();
  }, [activeCity]);

  // Format data for chart
  const chartData = forecastData?.timeline?.map(item => {
    const date = new Date(item.timestamp);
    return {
      time: date.toLocaleString('en-US', { weekday: 'short', hour: '2-digit' }),
      actual: item.actual_aqi,
      predicted: item.predicted_aqi,
      lower_bound: item.predicted_aqi ? Math.max(0, item.predicted_aqi - item.uncertainty) : null,
      upper_bound: item.predicted_aqi ? item.predicted_aqi + item.uncertainty : null,
    };
  }) || [];

  return (
    <main className="predictions-page">
      <header className="page-header">
        <h1>AQI Forecasts</h1>
        <p className="subtitle">AI-powered 7-day air quality predictions for major cities</p>
      </header>

      {/* City Tabs */}
      <div className="city-tabs">
        {TARGET_CITIES.map((city) => (
          <button
            key={city.slug}
            className={`tab-btn ${activeCity === city.slug ? "active" : ""}`}
            onClick={() => setActiveCity(city.slug)}
          >
            {city.name}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="loading">Loading forecast model...</div>
      ) : error ? (
        <div className="error">Error loading predictions: {error}</div>
      ) : forecastData ? (
        <div className="dashboard-content">
          {/* Top Stats */}
          <section className="top-stats">
            <div className="stat-card">
              <h3>Current AQI</h3>
              <div className="value">{forecastData.current.aqi.toFixed(0)}</div>
              <div className="sub">Latest Observation</div>
            </div>
            
            <div className="stat-card highlight">
              <h3>24h Forecast</h3>
              <div className="value">
                {forecastData.forecast_summary.next_24h.toFixed(0)}
              </div>
              <div className="sub">
                {forecastData.forecast_summary.trend === 'up' ? '📈 Worsening' : '📉 Improving'}
              </div>
            </div>

            <div className="stat-card model-info">
              <h3>Active Model</h3>
              <div className="value">{forecastData.model.name}</div>
              <div className="sub">Confidence: {(forecastData.model.confidence * 100).toFixed(1)}%</div>
            </div>
          </section>

          {/* Main Chart */}
          <section className="chart-container">
            <h2>7-Day Prediction Horizon</h2>
            <div style={{ width: '100%', height: 400 }}>
              <ResponsiveContainer>
                <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="time" tick={{ fill: '#9ca3af' }} />
                  <YAxis 
                    label={{ value: 'US AQI', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                    tick={{ fill: '#9ca3af' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px', color: '#fff' }}
                  />
                  <Legend />
                  
                  {/* Confidence Interval Area */}
                  <Area 
                    type="monotone" 
                    dataKey="upper_bound" 
                    stroke="none" 
                    fill="#3b82f6" 
                    fillOpacity={0.1}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="lower_bound" 
                    stroke="none" 
                    fill="#1f2937" 
                    fillOpacity={1} // Masks the area below lower bound
                  />

                  {/* Actual AQI Line */}
                  <Line 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#10b981" 
                    strokeWidth={3}
                    dot={{ r: 4 }}
                    name="Actual AQI"
                  />

                  {/* Predicted AQI Line */}
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    strokeDasharray="5 5"
                    dot={false}
                    name="Predicted AQI"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* AI Advisory */}
          <section className="ai-advisory">
            <h2>🤖 AI Insights</h2>
            <p>{forecastData.insights}</p>
          </section>
        </div>
      ) : null}

      <style jsx>{`
        .predictions-page {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }
        
        .page-header h1 {
          font-size: 2.5rem;
          color: var(--text);
          margin-bottom: 5px;
        }
        
        .subtitle {
          color: var(--muted);
          margin-bottom: 30px;
        }

        .city-tabs {
          display: flex;
          gap: 15px;
          margin-bottom: 30px;
          border-bottom: 1px solid var(--panel-border);
          padding-bottom: 10px;
          overflow-x: auto;
        }

        .tab-btn {
          background: transparent;
          border: none;
          color: var(--muted);
          font-size: 1.1rem;
          padding: 10px 20px;
          cursor: pointer;
          border-radius: 8px;
          transition: all 0.2s;
          white-space: nowrap;
        }

        .tab-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          color: var(--text);
        }

        .tab-btn.active {
          background: var(--accent);
          color: var(--bg);
          font-weight: 600;
        }

        .top-stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
          margin-bottom: 30px;
        }

        .stat-card {
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-radius: 12px;
          padding: 20px;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          box-shadow: var(--shadow);
          backdrop-filter: blur(16px);
        }

        .stat-card h3 {
          margin: 0 0 10px 0;
          color: var(--muted);
          font-size: 1rem;
          text-transform: uppercase;
          letter-spacing: 1px;
        }

        .stat-card .value {
          font-size: 2.5rem;
          font-weight: bold;
          color: var(--text);
          margin-bottom: 5px;
        }

        .stat-card .sub {
          color: var(--muted);
          font-size: 0.9rem;
          opacity: 0.8;
        }

        .stat-card.highlight {
          border-color: var(--accent);
          background: rgba(119, 228, 247, 0.08);
        }

        .stat-card.highlight .value {
          color: var(--accent);
        }

        .chart-container {
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-radius: 12px;
          padding: 25px;
          margin-bottom: 30px;
          box-shadow: var(--shadow);
          backdrop-filter: blur(16px);
        }

        .chart-container h2 {
          margin: 0 0 20px 0;
          color: var(--text);
        }

        .ai-advisory {
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-left: 4px solid var(--accent-2);
          border-radius: 12px;
          padding: 25px;
          box-shadow: var(--shadow);
          backdrop-filter: blur(16px);
        }

        .ai-advisory h2 {
          margin: 0 0 15px 0;
          color: var(--accent-2);
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .ai-advisory p {
          color: var(--text);
          line-height: 1.6;
          margin: 0;
          font-size: 1.1rem;
        }

        .loading, .error {
          text-align: center;
          padding: 50px;
          color: var(--muted);
          font-size: 1.2rem;
          background: var(--panel);
          border-radius: 12px;
          border: 1px solid var(--panel-border);
        }

        @media (max-width: 768px) {
          .top-stats {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </main>
  );
}
