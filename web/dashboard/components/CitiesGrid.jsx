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
        <h1>Air Quality Index — All Cities</h1>
        <p className="subtitle">Real-time AQI for {sortedCities.length} global cities</p>
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
          padding: 20px;
          font-family: system-ui, -apple-system, sans-serif;
        }

        .page-header {
          margin-bottom: 30px;
        }

        .page-header h1 {
          font-size: 2rem;
          margin: 0;
          color: #1f2937;
        }

        .page-header .subtitle {
          margin: 8px 0 0 0;
          color: #6b7280;
          font-size: 1rem;
        }

        .stats-cards {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin-bottom: 30px;
        }

        .stat-card {
          padding: 20px;
          border-radius: 8px;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          background: white;
          border-top: 4px solid;
        }

        .stat-card.worst {
          border-top-color: #7c2d12;
        }

        .stat-card.best {
          border-top-color: #10b981;
        }

        .stat-card.average {
          border-top-color: #3b82f6;
        }

        .stat-card h3 {
          margin: 0 0 8px 0;
          font-size: 0.875rem;
          color: #6b7280;
          font-weight: 600;
        }

        .stat-card .city-name {
          font-size: 1.25rem;
          font-weight: 600;
          color: #1f2937;
          margin: 8px 0;
        }

        .stat-card .value {
          font-size: 1.75rem;
          font-weight: bold;
          color: #374151;
          margin: 8px 0;
        }

        .stat-card .country,
        .stat-card .subtext {
          margin: 0;
          color: #9ca3af;
          font-size: 0.875rem;
        }

        .controls {
          display: flex;
          gap: 10px;
          margin-bottom: 20px;
        }

        .sort-buttons {
          display: flex;
          gap: 10px;
        }

        .sort-buttons button {
          padding: 8px 16px;
          border: 1px solid #d1d5db;
          background: white;
          border-radius: 6px;
          cursor: pointer;
          font-size: 0.875rem;
          font-weight: 500;
          transition: all 0.2s;
        }

        .sort-buttons button:hover {
          background: #f3f4f6;
        }

        .sort-buttons button.active {
          background: #3b82f6;
          color: white;
          border-color: #3b82f6;
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
          border: 1px solid #e5e7eb;
          border-left: 4px solid;
          border-radius: 8px;
          padding: 16px;
          background: white;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
          cursor: pointer;
        }

        .city-card:hover {
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
          transform: translateY(-4px);
        }

        .city-card .rank {
          position: absolute;
          top: 10px;
          right: 12px;
          background: #f3f4f6;
          width: 28px;
          height: 28px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 600;
          font-size: 0.875rem;
          color: #6b7280;
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
          color: #1f2937;
        }

        .city-card .body {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .city-card .country {
          margin: 0;
          color: #6b7280;
          font-size: 0.875rem;
        }

        .city-card .main-value {
          display: flex;
          align-items: baseline;
          gap: 8px;
        }

        .city-card .main-value .aqi {
          font-size: 2rem;
          font-weight: bold;
          color: #1f2937;
        }

        .city-card .main-value .label {
          color: #6b7280;
          font-size: 0.875rem;
        }

        .city-card .category {
          display: inline-block;
          padding: 4px 12px;
          background: #f3f4f6;
          border-radius: 12px;
          font-size: 0.75rem;
          font-weight: 600;
          text-transform: uppercase;
          color: #374151;
          width: fit-content;
        }

        .city-card .metrics {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: 8px;
          padding: 8px 0;
          border-top: 1px solid #f0f0f0;
          border-bottom: 1px solid #f0f0f0;
        }

        .city-card .metric {
          text-align: center;
        }

        .city-card .metric .name {
          display: block;
          font-size: 0.7rem;
          color: #9ca3af;
          text-transform: uppercase;
          margin-bottom: 2px;
        }

        .city-card .metric .value {
          display: block;
          font-size: 0.875rem;
          font-weight: 600;
          color: #1f2937;
        }

        .city-card .updated {
          color: #d1d5db;
          font-size: 0.75rem;
        }

        .city-card .hover-action {
          position: absolute;
          bottom: 10px;
          right: 16px;
          color: #3b82f6;
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
          padding: 40px 20px;
          text-align: center;
          font-size: 1rem;
          color: #374151;
        }

        @media (max-width: 768px) {
          .grid {
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
          }

          .stats-cards {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </main>
  );
}
