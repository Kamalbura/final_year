/**
 * Component: Single City Dashboard Page
 * Route: /dashboard/city/[citySlug]
 * 
 * Shows:
 * - Latest AQI + health category
 * - 24-hour trend line chart
 * - Pollutant breakdown (all 7 metrics)
 * - Health advisory based on AQI
 * - Daily/weekly comparison
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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

export default function CityDashboard({ citySlug }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState("24h");

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `/api/observations/${citySlug}?timeRange=${timeRange}`
        );
        if (!response.ok) throw new Error("Failed to fetch observation data");
        const result = await response.json();
        setData(result);
        setError(null);
      } catch (err) {
        setError(err.message);
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [citySlug, timeRange]);

  if (loading)
    return (
      <div className="loading">
        <span>Loading AQ data...</span>
      </div>
    );
  if (error) return <div className="error">Error: {error}</div>;
  if (!data) return <div className="error">No data available</div>;

  const { city, latest, hourly_stats, trend, aqi_category, health_advisory } =
    data;
  const formatNumber = (value, digits = 1) => Number(value ?? 0).toFixed(digits);

  // Format chart data
  const chartData = hourly_stats.map((stat) => ({
    time: new Date(stat.hour).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    }),
    aqi: stat.us_aqi_avg,
    pm2_5: stat.pm2_5_avg,
    pm10: stat.pm10_avg,
  }));

  // Format pollutants for radar
  const pollutantData = [
    {
      name: "PM2.5",
      value: Math.min(latest.pm2_5 || 0, 100),
      max: 100,
    },
    { name: "PM10", value: Math.min(latest.pm10 || 0, 100), max: 100 },
    { name: "NO2", value: Math.min(latest.nitrogen_dioxide || 0, 100), max: 100 },
    { name: "O3", value: Math.min(latest.ozone || 0, 100), max: 100 },
    { name: "SO2", value: Math.min(latest.sulphur_dioxide || 0, 100), max: 100 },
    { name: "CO", value: Math.min((latest.carbon_monoxide || 0) * 100, 100), max: 100 },
  ];

  const trendIcon =
    trend.direction === "increasing"
      ? "📈"
      : trend.direction === "decreasing"
        ? "📉"
        : "➡️";

  return (
    <main className="city-dashboard">
      {/* Header */}
      <section className="header">
        <div className="title-section">
          <h1>{city.name}</h1>
          <p className="location">{city.country}</p>
        </div>

        <div className="current-status" style={{ borderColor: aqi_category.color }}>
          <div className="aqi-display">
            <span className="emoji">{aqi_category.emoji}</span>
            <span className="value">{latest.us_aqi.toFixed(0)}</span>
            <span className="label">AQI</span>
          </div>
          <div className="status">
            <span
              className="category"
              style={{ backgroundColor: aqi_category.color }}
            >
              {aqi_category.value.replace(/_/g, " ").toUpperCase()}
            </span>
            <span className="pm2-5">PM2.5: {latest.pm2_5.toFixed(1)} µg/m³</span>
            <span className="pm10">PM10: {latest.pm10.toFixed(1)} µg/m³</span>
          </div>
        </div>

        <div className="trend">
          <span className="icon">{trendIcon}</span>
          <span>
            {trend.direction === "increasing"
              ? `Up ${Math.abs(trend.change_percent).toFixed(1)}%`
              : trend.direction === "decreasing"
                ? `Down ${Math.abs(trend.change_percent).toFixed(1)}%`
                : "Stable"}{" "}
            ({formatNumber(trend.change_absolute, 1)} AQI)
          </span>
        </div>
      </section>

      {/* Time Range Selector */}
      <section className="time-range-selector">
        {["24h", "7d", "30d"].map((range) => (
          <button
            key={range}
            className={timeRange === range ? "active" : ""}
            onClick={() => setTimeRange(range)}
          >
            {range === "24h" ? "24 Hours" : range === "7d" ? "7 Days" : "30 Days"}
          </button>
        ))}
      </section>

      {/* Chart */}
      <section className="chart-section">
        <h2>AQI Trend</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis yAxisId="left" label={{ value: "AQI", angle: -90, position: "insideLeft" }} />
            <YAxis
              yAxisId="right"
              orientation="right"
              label={{ value: "µg/m³", angle: 90, position: "insideRight" }}
            />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="aqi"
              stroke={aqi_category.color}
              strokeWidth={2}
              name="AQI"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="pm2_5"
              stroke="#8b5cf6"
              name="PM2.5"
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="pm10"
              stroke="#06b6d4"
              name="PM10"
            />
          </LineChart>
        </ResponsiveContainer>
      </section>

      {/* Pollutant Breakdown */}
      <section className="pollutants-section">
        <div className="left">
          <h2>Pollutant Breakdown</h2>
          <div className="pollutant-list">
            <div className="pollutant-item">
              <span className="label">PM2.5</span>
              <span className="value">{formatNumber(latest.pm2_5, 1)}</span>
              <span className="unit">µg/m³</span>
            </div>
            <div className="pollutant-item">
              <span className="label">PM10</span>
              <span className="value">{formatNumber(latest.pm10, 1)}</span>
              <span className="unit">µg/m³</span>
            </div>
            <div className="pollutant-item">
              <span className="label">NO2</span>
              <span className="value">{formatNumber(latest.nitrogen_dioxide, 1)}</span>
              <span className="unit">ppb</span>
            </div>
            <div className="pollutant-item">
              <span className="label">O3</span>
              <span className="value">{formatNumber(latest.ozone, 1)}</span>
              <span className="unit">ppb</span>
            </div>
            <div className="pollutant-item">
              <span className="label">SO2</span>
              <span className="value">{formatNumber(latest.sulphur_dioxide, 1)}</span>
              <span className="unit">ppb</span>
            </div>
            <div className="pollutant-item">
              <span className="label">CO</span>
              <span className="value">{formatNumber(latest.carbon_monoxide, 2)}</span>
              <span className="unit">ppm</span>
            </div>
          </div>
        </div>

        <div className="right">
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={pollutantData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="name" />
              <PolarRadiusAxis angle={90} domain={[0, 100]} />
              <Radar name="Pollutant Level" dataKey="value" stroke={aqi_category.color} fill={aqi_category.color} fillOpacity={0.6} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Health Advisory */}
      <section className="advisory-section" style={{ borderLeft: `4px solid ${aqi_category.color}` }}>
        <h2>🏥 Health Advisory</h2>
        <div className="advisory-content">
          <div className="group">
            <h3>General Population</h3>
            <p>{health_advisory.general}</p>
          </div>
          <div className="group">
            <h3>Sensitive Groups</h3>
            <p>{health_advisory.sensitive_groups}</p>
          </div>
          <div className="group">
            <h3>Children</h3>
            <p>{health_advisory.children}</p>
          </div>
        </div>
      </section>

      {/* Last Updated */}
      <footer className="footer">
        <small>Last updated: {new Date(latest.observed_at).toLocaleString()}</small>
      </footer>

      <style jsx>{`
        .city-dashboard {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
          font-family: system-ui, -apple-system, sans-serif;
        }

        .header {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 20px;
          margin-bottom: 30px;
          align-items: center;
        }

        .title-section h1 {
          font-size: 2.5rem;
          margin: 0;
          color: #1f2937;
        }

        .title-section .location {
          margin: 5px 0 0 0;
          color: #6b7280;
        }

        .current-status {
          border: 3px solid;
          border-radius: 12px;
          padding: 20px;
          background: #f9fafb;
          display: flex;
          align-items: center;
          gap: 20px;
        }

        .aqi-display {
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .aqi-display .emoji {
          font-size: 2rem;
        }

        .aqi-display .value {
          font-size: 2.5rem;
          font-weight: bold;
          color: #1f2937;
        }

        .aqi-display .label {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .status {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .status .category {
          padding: 6px 12px;
          border-radius: 6px;
          color: white;
          font-weight: 600;
          font-size: 0.875rem;
          display: inline-block;
          width: fit-content;
        }

        .status span {
          color: #374151;
          font-size: 0.875rem;
        }

        .trend {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 1.125rem;
          font-weight: 600;
        }

        .trend .icon {
          font-size: 1.5rem;
        }

        .time-range-selector {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
        }

        .time-range-selector button {
          padding: 8px 16px;
          border: 1px solid #d1d5db;
          background: white;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.875rem;
          font-weight: 500;
          transition: all 0.2s;
        }

        .time-range-selector button:hover {
          background: #f3f4f6;
        }

        .time-range-selector button.active {
          background: #3b82f6;
          color: white;
          border-color: #3b82f6;
        }

        .chart-section {
          background: white;
          padding: 20px;
          border-radius: 8px;
          margin-bottom: 30px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .chart-section h2 {
          margin: 0 0 20px 0;
          color: #1f2937;
        }

        .pollutants-section {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 20px;
          margin-bottom: 30px;
        }

        .pollutants-section h2 {
          grid-column: 1;
          margin: 0 0 10px 0;
          color: #1f2937;
        }

        .pollutant-list {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          display: grid;
          gap: 12px;
        }

        .pollutant-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 0;
          border-bottom: 1px solid #e5e7eb;
        }

        .pollutant-item:last-child {
          border-bottom: none;
        }

        .pollutant-item .label {
          font-weight: 600;
          color: #1f2937;
          min-width: 50px;
        }

        .pollutant-item .value {
          font-size: 1.25rem;
          font-weight: bold;
          color: #3b82f6;
        }

        .pollutant-item .unit {
          font-size: 0.75rem;
          color: #6b7280;
          margin-left: 8px;
        }

        .pollutants-section .right {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .advisory-section {
          background: white;
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          margin-bottom: 30px;
        }

        .advisory-section h2 {
          margin: 0 0 20px 0;
          color: #1f2937;
        }

        .advisory-content {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 20px;
        }

        .advisory-content .group h3 {
          margin: 0 0 10px 0;
          font-size: 1rem;
          color: #374151;
        }

        .advisory-content .group p {
          margin: 0;
          color: #4b5563;
          line-height: 1.5;
          font-size: 0.875rem;
        }

        .footer {
          text-align: center;
          color: #9ca3af;
          font-size: 0.875rem;
          padding-top: 20px;
        }

        .loading,
        .error {
          padding: 20px;
          text-align: center;
          font-size: 1rem;
          color: #374151;
        }

        @media (max-width: 768px) {
          .header {
            grid-template-columns: 1fr;
          }

          .pollutants-section {
            grid-template-columns: 1fr;
          }

          .advisory-content {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </main>
  );
}
