import pool from "../../lib/db-pool.js";

export async function GET(request, { params }) {
  const { citySlug } = await params;
  
  try {
    // Get past 24 hours of actual data from materialized view
    const pastResult = await pool.query(
      `
      SELECT
        hour as timestamp,
        us_aqi_avg as actual_aqi
      FROM aq.hourly_aggregates
      WHERE city_slug = $1
        AND hour >= NOW() - INTERVAL '24 hours'
      ORDER BY hour ASC
      `,
      [citySlug]
    );

    // Get forecasts for the future from aq.forecasts
    const futureResult = await pool.query(
      `
      SELECT
        horizon_timestamp as timestamp,
        predicted_us_aqi as predicted_aqi,
        confidence,
        model_type
      FROM aq.forecasts
      WHERE city_slug = $1
        AND horizon_timestamp >= NOW()
      ORDER BY horizon_timestamp ASC
      LIMIT 168
      `,
      [citySlug]
    );

    const timeline = [];
    
    // Add past actuals
    for (const row of pastResult.rows) {
      timeline.push({
        timestamp: row.timestamp,
        actual_aqi: row.actual_aqi ? Math.round(row.actual_aqi) : null,
        predicted_aqi: null,
        uncertainty: 0
      });
    }

    // Add future predictions
    let currentModel = "No Model Available";
    let modelConfidence = 0;
    
    for (const row of futureResult.rows) {
      const uncertainty = row.confidence ? (1 - row.confidence) * 50 : 20; // estimate uncertainty
      timeline.push({
        timestamp: row.timestamp,
        actual_aqi: null,
        predicted_aqi: Math.round(row.predicted_aqi),
        uncertainty: Math.round(uncertainty)
      });
      currentModel = row.model_type || currentModel;
      modelConfidence = row.confidence || modelConfidence;
    }

    const currentActual = timeline.filter(t => t.actual_aqi !== null).pop()?.actual_aqi || 0;
    const futureData = timeline.filter(t => t.predicted_aqi !== null);
    
    const next24hAvg = futureData.length > 0 
      ? futureData.slice(0, 4).reduce((sum, t) => sum + t.predicted_aqi, 0) / Math.min(4, futureData.length)
      : 0;

    let trend = 'stable';
    if (next24hAvg > currentActual + 5) trend = 'up';
    if (next24hAvg < currentActual - 5) trend = 'down';

    const data = {
      city: citySlug,
      current: {
        aqi: currentActual,
        timestamp: new Date().toISOString()
      },
      forecast_summary: {
        next_24h: next24hAvg,
        trend: trend
      },
      model: {
        name: futureData.length > 0 ? currentModel : "Awaiting Training",
        confidence: futureData.length > 0 ? modelConfidence : 0,
      },
      insights: futureData.length > 0 
        ? `Model inference complete. Trend is ${trend} over the next 24 hours.`
        : "No forecast data available yet. Please run the model training pipeline to generate predictions.",
      timeline: timeline
    };

    return new Response(JSON.stringify(data), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error fetching predictions:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}

