import fs from 'fs';
import path from 'path';

// Models available with their metadata
const MODELS = {
  "xgboost":       { tier: 1, family: "Tree Ensemble",   name: "XGBoost",       category: "Best" },
  "lightgbm":      { tier: 1, family: "Tree Ensemble",   name: "LightGBM",      category: "Best" },
  "random_forest": { tier: 1, family: "Tree Ensemble",   name: "Random Forest", category: "Best" },
  "cnn_lstm":      { tier: 2, family: "DL Hybrid",       name: "CNN-LSTM",      category: "DL Best" },
  "gru":           { tier: 2, family: "DL Sequence",     name: "GRU",           category: "DL Best" },
  "transformer":   { tier: 2, family: "Transformer",     name: "Transformer",   category: "DL Best" },
  "svr":           { tier: 3, family: "Support Vector",  name: "SVR",           category: "Good" },
  "bilstm":        { tier: 3, family: "DL Sequence",     name: "BiLSTM",        category: "Good" },
};

const PREDICTIONS_DIR = process.env.DASHBOARD_PROJECT_ROOT
  ? `${process.env.DASHBOARD_PROJECT_ROOT}/data/predictions`
  : "/opt/final_year/data/predictions";

export async function GET(request, { params }) {
  const { citySlug } = await params;
  const { searchParams } = new URL(request.url);
  const modelKey = searchParams.get("model") || "xgboost";
  const modelInfo = MODELS[modelKey] || MODELS["xgboost"];
  
  try {
    // Read predictions from JSON file
    const predFile = path.join(PREDICTIONS_DIR, `${modelKey}_latest.json`);
    
    if (!fs.existsSync(predFile)) {
      return new Response(JSON.stringify({
        error: "No predictions yet",
        message: `Model ${modelInfo.name} hasn't generated predictions yet. They run hourly at :15 past.`,
        model: modelInfo,
        timeline: []
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    
    const data = JSON.parse(fs.readFileSync(predFile, 'utf-8'));
    const cityData = data.predictions?.[citySlug] || data.predictions?.[citySlug.replace('-', '_')];
    
    if (!cityData || !cityData.forecasts?.length) {
      return new Response(JSON.stringify({
        error: "No city data",
        message: `No predictions for ${citySlug} using ${modelInfo.name}.`,
        model: modelInfo,
        timeline: []
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    
    // Build timeline: current actual + future predictions
    const forecast = cityData.forecasts;
    const currentPm25 = forecast[0]?.pm2_5 || 0;
    
    const timeline = forecast.map(f => ({
      timestamp: f.timestamp,
      hour: f.hour,
      predicted_pm2_5: f.pm2_5,
      actual_aqi: null,
    }));
    
    // Add current value as first actual point
    timeline.unshift({
      timestamp: data.generated_at,
      hour: 0,
      predicted_pm2_5: null,
      actual_aqi: currentPm25,
    });
    
    // Compute trend
    const firstHalf = forecast.slice(0, 12);
    const secondHalf = forecast.slice(12, 24);
    const firstAvg = firstHalf.reduce((s, f) => s + f.pm2_5, 0) / Math.max(1, firstHalf.length);
    const secondAvg = secondHalf.reduce((s, f) => s + f.pm2_5, 0) / Math.max(1, secondHalf.length);
    const trend = secondAvg > firstAvg + 2 ? 'up' : secondAvg < firstAvg - 2 ? 'down' : 'stable';
    
    return new Response(JSON.stringify({
      city: citySlug,
      model: modelInfo,
      available_models: Object.entries(MODELS).map(([k, v]) => ({
        key: k,
        name: v.name,
        family: v.family,
        tier: v.tier,
        category: v.category
      })),
      generated_at: data.generated_at,
      current_pm2_5: currentPm25,
      forecast_summary: {
        next_12h: Math.round(firstAvg * 10) / 10,
        next_24h: Math.round((firstAvg + secondAvg) / 2 * 10) / 10,
        trend,
        trend_label: trend === 'up' ? 'Worsening' : trend === 'down' ? 'Improving' : 'Stable'
      },
      insights: `PM2.5 forecast by ${modelInfo.name} (${modelInfo.family}). Trend: ${trend === 'up' ? 'worsening over next 24h' : trend === 'down' ? 'improving over next 24h' : 'stable'}. Current: ${currentPm25} µg/m³.`,
      timeline
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
    
  } catch (error) {
    console.error("Prediction API error:", error);
    return new Response(JSON.stringify({
      error: error.message,
      model: modelInfo,
      timeline: []
    }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
