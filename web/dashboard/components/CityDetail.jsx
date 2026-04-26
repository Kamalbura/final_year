'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import styles from './CityDetail.module.css';

const AQI_CATEGORIES = {
  1: { label: 'Good', min: 0, max: 50, color: '#10b981', advice: 'Air quality is satisfactory for outdoor activities.' },
  2: { label: 'Moderate', min: 51, max: 100, color: '#f59e0b', advice: 'Sensitive individuals may experience minor symptoms.' },
  3: { label: 'Unhealthy for Sensitive Groups', min: 101, max: 150, color: '#f97316', advice: 'Members of sensitive groups (children, elderly) should limit prolonged outdoor activities.' },
  4: { label: 'Unhealthy', min: 151, max: 200, color: '#ef4444', advice: 'General population may begin to experience health effects. Limit outdoor activities.' },
  5: { label: 'Very Unhealthy', min: 201, max: 300, color: '#8b5cf6', advice: 'Health alert. Entire population may experience severe health effects.' },
  6: { label: 'Hazardous', min: 301, max: 500, color: '#7f1d1d', advice: 'Health warning. Avoid outdoor activities. Remain indoors.' },
};

function getAQICategory(aqi) {
  for (const [key, category] of Object.entries(AQI_CATEGORIES)) {
    if (aqi >= category.min && aqi <= category.max) {
      return category;
    }
  }
  return AQI_CATEGORIES[6];
}

function getTrend(current, previous) {
  if (!previous) return null;
  const diff = current - previous;
  if (diff > 5) return { direction: '▲', label: 'Worsening', color: '#ef4444' };
  if (diff < -5) return { direction: '▼', label: 'Improving', color: '#10b981' };
  return { direction: '→', label: 'Stable', color: '#f59e0b' };
}

export default function CityDetail({ params }) {
  const { citySlug } = params;
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState('24h');

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/observations/${citySlug}?timeRange=${timeRange}`);
        if (!response.ok) throw new Error('Failed to fetch city data');
        const cityData = await response.json();
        setData(cityData);
      } catch (err) {
        setError(err.message);
        console.error('Error fetching city data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, [citySlug, timeRange]);

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loader}>Loading city data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.container}>
        <Link href="/dashboard/cities" className={styles.backLink}>
          ← Back to All Cities
        </Link>
        <div className={styles.error}>Error: {error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className={styles.container}>
        <Link href="/dashboard/cities" className={styles.backLink}>
          ← Back to All Cities
        </Link>
        <div className={styles.error}>No data available for this city</div>
      </div>
    );
  }

  const category = getAQICategory(data.aqi_index || 0);
  const trend = getTrend(data.aqi_index, data.previous_aqi_index);

  return (
    <div className={styles.container}>
      <Link href="/dashboard/cities" className={styles.backLink}>
        ← Back to All Cities
      </Link>

      <div className={styles.header} style={{ borderColor: category.color }}>
        <div>
          <h1>{data.city}, {data.country || 'Unknown'}</h1>
          <div className={styles.liveIndicator}>
            <span className={styles.liveDot}></span> Live Data
          </div>
        </div>
        <button className={styles.refreshBtn} onClick={() => window.location.reload()}>
          ⟳ Refresh
        </button>
      </div>

      <div className={styles.primaryMetrics}>
        <div className={styles.aqiBox} style={{ backgroundColor: category.color }}>
          <div className={styles.aqiValue}>{data.aqi_index || 'N/A'}</div>
          <div className={styles.aqiLabel}>AQI</div>
          <div className={styles.aqiCategory}>{category.label}</div>
          {trend && (
            <div className={styles.trend} style={{ color: trend.color }}>
              {trend.direction} {trend.label}
            </div>
          )}
        </div>

        <div className={styles.metricsGrid}>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>PM2.5</span>
            <span className={styles.metricValue}>{(data.pm25 || 0).toFixed(1)}</span>
            <span className={styles.metricUnit}>µg/m³</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>PM10</span>
            <span className={styles.metricValue}>{(data.pm10 || 0).toFixed(1)}</span>
            <span className={styles.metricUnit}>µg/m³</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>NO2</span>
            <span className={styles.metricValue}>{(data.no2 || 0).toFixed(1)}</span>
            <span className={styles.metricUnit}>ppb</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>O3</span>
            <span className={styles.metricValue}>{(data.o3 || 0).toFixed(1)}</span>
            <span className={styles.metricUnit}>ppb</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>SO2</span>
            <span className={styles.metricValue}>{(data.so2 || 0).toFixed(1)}</span>
            <span className={styles.metricUnit}>ppb</span>
          </div>
          <div className={styles.metric}>
            <span className={styles.metricLabel}>CO</span>
            <span className={styles.metricValue}>{(data.co || 0).toFixed(1)}</span>
            <span className={styles.metricUnit}>ppm</span>
          </div>
        </div>
      </div>

      <div className={styles.healthAdvisory} style={{ backgroundColor: `${category.color}20`, borderLeft: `4px solid ${category.color}` }}>
        <h3>🏥 Health Advisory</h3>
        <p>{category.advice}</p>
      </div>

      <div className={styles.timeRangeControls}>
        <label>Time Range:</label>
        <div className={styles.buttons}>
          <button className={timeRange === '24h' ? styles.active : ''} onClick={() => setTimeRange('24h')}>
            24 Hours
          </button>
          <button className={timeRange === '7d' ? styles.active : ''} onClick={() => setTimeRange('7d')}>
            7 Days
          </button>
          <button className={timeRange === '30d' ? styles.active : ''} onClick={() => setTimeRange('30d')}>
            30 Days
          </button>
        </div>
      </div>

      <div className={styles.chartPlaceholder}>
        <p>📊 AQI Trend Chart ({timeRange})</p>
        <p style={{ fontSize: '12px', color: '#6b7280' }}>Chart visualization will be displayed here</p>
      </div>

      <div className={styles.lastUpdated}>Last updated: {new Date(data.timestamp).toLocaleString()}</div>
    </div>
  );
}
