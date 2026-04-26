/**
 * Component: All Cities Ranking Grid
 * Route: /dashboard/cities
 * 
 * Shows:
 * - All 36 cities with current AQI
 * - Sorted by worst AQI first (descending)
 * - Color-coded by category
 * - Quick links to individual city dashboard
 */

"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";

export default function CitiesGrid() {
  const [cities, setCities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sortBy, setSortBy] = useState("aqi"); // aqi, name, pm2_5

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch("/api/observations");
        if (!response.ok) throw new Error("Failed to fetch cities");
        const result = await response.json();
        setCities(result);
        setError(null);
      } catch (err) {
        setError(err.message);
        setCities([]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const sortedCities = [...cities].sort((a, b) => {
    switch (sortBy) {
      case "pm2_5":
        return (b.pm2_5 || 0) - (a.pm2_5 || 0);
      case "name":
        return a.city.name.localeCompare(b.city.name);
      case "aqi":
      default:
        return (b.current_aqi || 0) - (a.current_aqi || 0);
    }
  });

  if (loading)
    return (
      <div className="loading">
        <span>Loading cities...</span>
      </div>
    );
  if (error) return <div className="error">Error: {error}</div>;

  if (sortedCities.length === 0)
    return (
      <main className="cities-grid-page">
        <header className="page-header">
          <h1>Air Quality Index - All Cities</h1>
          <p className="subtitle">No observation records are currently available.</p>
        </header>
      </main>
    );

  const stats = {
    best: sortedCities[sortedCities.length - 1],
    worst: sortedCities[0],
    avg: (sortedCities.reduce((sum, c) => sum + (c.current_aqi || 0), 0) / sortedCities.length).toFixed(1),
  };

  return (
    <main className="cities-grid-page">
      <header className="page-header">
        <h1 style={{ color: "#eaf2ff", textShadow: "0 1px 2px rgba(0, 0, 0, 0.35)" }}>
          Air Quality Index — All Cities
        </h1>
        <p
          className="subtitle"
          style={{ color: "#d7e2f0", textShadow: "0 1px 1px rgba(0, 0, 0, 0.25)" }}
        >
          Real-time AQI for {sortedCities.length} global cities
        </p>
      </header>

      {/* Statistics Cards */}
      <section className="stats-cards">
        <article className="stat-card worst">
          <h3>🔴 Worst Air Quality</h3>
          <p className="city-name">{stats.worst.city.name}</p>
          <p className="value">{stats.worst.current_aqi.toFixed(0)} AQI</p>
          <p className="country">{stats.worst.city.country}</p>
        </article>

        <article className="stat-card best">
          <h3>✅ Best Air Quality</h3>
          <p className="city-name">{stats.best.city.name}</p>
          <p className="value">{stats.best.current_aqi.toFixed(0)} AQI</p>
          <p className="country">{stats.best.city.country}</p>
        </article>

        <article className="stat-card average">
          <h3>📊 Global Average</h3>
          <p className="value">{stats.avg} AQI</p>
          <p className="subtext">Across all {sortedCities.length} cities</p>
        </article>
      </section>

      {/* Sorting Controls */}
      <section className="controls">
        <div className="sort-buttons">
          {["aqi", "pm2_5", "name"].map((sort) => (
            <button
              key={sort}
              className={sortBy === sort ? "active" : ""}
              onClick={() => setSortBy(sort)}
            >
              {sort === "aqi"
                ? "Sort by AQI"
                : sort === "pm2_5"
                  ? "Sort by PM2.5"
                  : "Sort by Name"}
            </button>
          ))}
        </div>
      </section>

      {/* Cities Grid */}
      <section className="grid">
        {sortedCities.map((city, idx) => (
          <Link key={city.city.slug} href={`/air-quality/city/${city.city.slug}`}>
            <article
              className="city-card"
              style={{
                borderLeftColor: city.color,
                backgroundColor:
                  city.color + "15", /* Add transparency */
              }}
            >
              <div className="rank">{idx + 1}</div>

              <div className="header">
                <span className="emoji">{city.emoji}</span>
                <h3>{city.city.name}</h3>
              </div>

              <div className="body">
                <p className="country">{city.city.country}</p>

                <div className="main-value">
                  <span className="aqi">{city.current_aqi.toFixed(0)}</span>
                  <span className="label">AQI</span>
                </div>

                <span className="category">{city.category.replace(/_/g, " ")}</span>

                <div className="metrics">
                  <div className="metric">
                    <span className="name">PM2.5</span>
                    <span className="value">{city.pm2_5.toFixed(1)}</span>
                  </div>
                  <div className="metric">
                    <span className="name">PM10</span>
                    <span className="value">{city.pm10.toFixed(1)}</span>
                  </div>
                  <div className="metric">
                    <span className="name">NO2</span>
                    <span className="value">{city.no2.toFixed(1)}</span>
                  </div>
                </div>

                <small className="updated">
                  Updated: {new Date(city.observed_at).toLocaleTimeString()}
                </small>
              </div>

              <div className="hover-action">View Details →</div>
            </article>
          </Link>
        ))}
      </section>

      <style jsx>{`
        .cities-grid-page {
          max-width: 1400px;
          margin: 0 auto;
          padding: 28px 20px 56px;
        }

        .page-header {
          margin-bottom: 26px;
        }

        .page-header h1 {
          font-size: clamp(2rem, 4vw, 3.2rem);
          line-height: 0.98;
          margin: 0;
          color: var(--text);
          letter-spacing: -0.04em;
        }

        .page-header .subtitle {
          margin: 10px 0 0 0;
          color: var(--muted);
          font-size: 1rem;
          max-width: 62ch;
        }

        .stats-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 16px;
          margin-bottom: 22px;
        }

        .stat-card {
          padding: 20px;
          border-radius: 22px;
          background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: var(--shadow);
          backdrop-filter: blur(16px);
          position: relative;
          overflow: hidden;
        }

        .stat-card.worst {
          border-color: rgba(249, 115, 22, 0.28);
        }

        .stat-card.best {
          border-color: rgba(38, 208, 124, 0.28);
        }

        .stat-card.average {
          border-color: rgba(119, 228, 247, 0.28);
        }

        .stat-card h3 {
          margin: 0 0 8px 0;
          font-size: 0.875rem;
          color: var(--muted);
          font-weight: 600;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }

        .stat-card .city-name {
          font-size: 1.25rem;
          font-weight: 600;
          color: var(--text);
          margin: 8px 0;
        }

        .stat-card .value {
          font-size: 2rem;
          font-weight: 800;
          color: var(--text);
          margin: 8px 0;
        }

        .stat-card .country,
        .stat-card .subtext {
          margin: 0;
          color: var(--muted);
          font-size: 0.875rem;
        }

        .controls {
          display: flex;
          gap: 10px;
          margin-bottom: 18px;
          flex-wrap: wrap;
        }

        .sort-buttons {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }

        .sort-buttons button {
          padding: 10px 16px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.04);
          border-radius: 999px;
          cursor: pointer;
          font-size: 0.875rem;
          font-weight: 500;
          transition: all 0.2s;
          color: var(--text);
        }

        .sort-buttons button:hover {
          background: rgba(255, 255, 255, 0.08);
          transform: translateY(-1px);
        }

        .sort-buttons button.active {
          background: linear-gradient(135deg, var(--accent), var(--accent-2));
          color: white;
          border-color: transparent;
          box-shadow: 0 14px 30px rgba(119, 228, 247, 0.16);
        }

        .grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 16px;
        }

        .grid a {
          text-decoration: none;
          color: inherit;
        }

        .city-card {
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-left: 4px solid;
          border-radius: 24px;
          padding: 18px 18px 16px;
          background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03));
          box-shadow: var(--shadow);
          backdrop-filter: blur(18px);
          transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
          position: relative;
          overflow: hidden;
          cursor: pointer;
        }

        .city-card:hover {
          box-shadow: 0 28px 80px rgba(0, 0, 0, 0.42);
          transform: translateY(-4px);
          border-color: rgba(119, 228, 247, 0.22);
        }

        .city-card .rank {
          position: absolute;
          top: 10px;
          right: 12px;
          background: rgba(255, 255, 255, 0.08);
          width: 28px;
          height: 28px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 600;
          font-size: 0.875rem;
          color: var(--text);
        }

        .city-card .header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
        }

        .city-card .header .emoji {
          font-size: 1.5rem;
        }

        .city-card .header h3 {
          margin: 0;
          font-size: 1.125rem;
          color: var(--text);
        }

        .city-card .body {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .city-card .country {
          margin: 0;
          color: var(--muted);
          font-size: 0.875rem;
        }

        .city-card .main-value {
          display: flex;
          align-items: baseline;
          gap: 8px;
        }

        .city-card .main-value .aqi {
          font-size: 2rem;
          font-weight: 800;
          color: var(--text);
        }

        .city-card .main-value .label {
          color: var(--muted);
          font-size: 0.875rem;
        }

        .city-card .category {
          display: inline-block;
          padding: 5px 12px;
          background: rgba(119, 228, 247, 0.12);
          border: 1px solid rgba(119, 228, 247, 0.18);
          border-radius: 999px;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          color: #d9fbff;
          width: fit-content;
        }

        .city-card .metrics {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 8px;
          padding: 12px 0;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .city-card .metric {
          text-align: center;
        }

        .city-card .metric .name {
          display: block;
          font-size: 0.7rem;
          color: var(--muted);
          text-transform: uppercase;
          margin-bottom: 2px;
        }

        .city-card .metric .value {
          display: block;
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--text);
        }

        .city-card .updated {
          color: var(--muted);
          font-size: 0.75rem;
        }

        .city-card .hover-action {
          position: absolute;
          bottom: 10px;
          right: 16px;
          color: var(--accent);
          font-size: 0.875rem;
          font-weight: 600;
          opacity: 0;
          transition: opacity 0.3s;
        }

        .city-card:hover .hover-action {
          opacity: 1;
        }

        .loading,
        .error {
          padding: 48px 20px;
          text-align: center;
          font-size: 1rem;
          color: var(--muted);
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 18px;
          box-shadow: var(--shadow);
        }

        .error {
          color: #ffb3b3;
        }

        @media (max-width: 768px) {
          .grid {
            grid-template-columns: 1fr;
          }

          .stats-cards {
            grid-template-columns: 1fr;
          }

          .controls,
          .sort-buttons {
            width: 100%;
          }

          .sort-buttons button {
            flex: 1 1 auto;
          }

          .city-card .metrics {
            grid-template-columns: repeat(3, minmax(0, 1fr));
          }
        }
      `}</style>
    </main>
  );
}
