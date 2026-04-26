/**
 * API Route: GET /api/observations/:citySlug
 * 
 * Returns hourly/daily aggregates + trend analysis for a city.
 * 
 * Query Parameters:
 *   - timeRange = "24h" | "7d" | "30d" (default: 24h)
 *   - metric = "pm2_5" | "pm10" | "us_aqi" | all (default: all)
 * 
 * Response:
 * {
 *   city: { slug, name, latitude, longitude },
 *   latest: { observed_at, us_aqi, pm2_5, pm10, ... all_metrics },
 *   hourly_stats: [ { hour, pm2_5_avg, pm2_5_max, ... }, ... ],
 *   daily_stats: [ { day, ... }, ... ],
 *   trend: { direction, slope, change_percent, change_absolute },
 *   aqi_category: "good|moderate|unhealthy_sg|unhealthy|very_unhealthy|hazardous",
 *   health_advisory: "..."
 * }
 */

import { ObservationRepository } from "../../../lib/observations_repo.js";
import { getSingleCity } from "../../../lib/cities.js";

const AQI_CATEGORIES = [
  { min: 0, max: 50, name: "good", color: "#10b981", emoji: "✅" },
  { min: 51, max: 100, name: "moderate", color: "#f59e0b", emoji: "🟡" },
  { min: 101, max: 150, name: "unhealthy_sg", color: "#f97316", emoji: "🟠" },
  { min: 151, max: 200, name: "unhealthy", color: "#ef4444", emoji: "🔴" },
  { min: 201, max: 300, name: "very_unhealthy", color: "#9333ea", emoji: "🟣" },
  { min: 301, max: 500, name: "hazardous", color: "#7c2d12", emoji: "🟤" },
];

const HEALTH_ADVISORIES = {
  good: {
    general:
      "Air quality is satisfactory, and air pollution poses little or no risk.",
    sensitive_groups: "Maintain normal activities.",
    children: "Safe for outdoor activities.",
  },
  moderate: {
    general:
      "Air quality is acceptable; however, there may be risk for some people, particularly those who are unusually sensitive to air pollution.",
    sensitive_groups:
      "Consider limiting prolonged outdoor exertion; keep windows closed during peak hours.",
    children: "Safe, but sensitive children should limit vigorous outdoor exertion.",
  },
  unhealthy_sg: {
    general:
      "Members of sensitive groups (children, elderly, people with respiratory disease) may experience health effects.",
    sensitive_groups:
      "Reduce prolonged outdoor exertion. Consider moving exertion indoors or rescheduling.",
    children:
      "Limit outdoor play. Move vigorous activities indoors or to another time.",
  },
  unhealthy: {
    general:
      "Some members of the general public may begin to experience health effects; members of sensitive groups may experience more serious health effects.",
    sensitive_groups:
      "Avoid outdoor exertion; stay indoors as much as possible.",
    children:
      "Avoid all outdoor activities. Stay inside in a well-ventilated building.",
  },
  very_unhealthy: {
    general:
      "Health alert: The risk of health effects is increased for the entire population.",
    sensitive_groups:
      "Health warning of emergency conditions: the entire population is more likely to be affected.",
    children:
      "Remain indoors; use air purifier if available. Avoid any exertion.",
  },
  hazardous: {
    general:
      "Health warning of emergency conditions: everyone is more likely to be affected by serious health effects.",
    sensitive_groups:
      "Everyone should avoid all outdoor exertion; remain indoors and keep activity levels low.",
    children:
      "Stay indoors. Use air purifier. Seek medical attention if symptoms develop.",
  },
};

function getAQICategory(aqi) {
  if (aqi === null || aqi === undefined) return null;
  for (const cat of AQI_CATEGORIES) {
    if (aqi >= cat.min && aqi <= cat.max) {
      return cat;
    }
  }
  return AQI_CATEGORIES[AQI_CATEGORIES.length - 1]; // Hazardous
}

function getHealthAdvisory(categoryName) {
  return HEALTH_ADVISORIES[categoryName] || HEALTH_ADVISORIES.moderate;
}

export async function GET(request, { params }) {
  try {
    const { citySlug } = params;
    const { searchParams } = new URL(request.url);

    const timeRange = searchParams.get("timeRange") || "24h";
    const metric = searchParams.get("metric") || "all";

    const hoursMap = { "24h": 24, "7d": 168, "30d": 720 };
    const hours = hoursMap[timeRange] || 24;

    const repo = new ObservationRepository();

    // Get city info
    const city = getSingleCity(citySlug);
    if (!city) {
      return new Response(
        JSON.stringify({ error: "City not found" }),
        { status: 404, headers: { "Content-Type": "application/json" } }
      );
    }

    // Get latest observation
    const latest = await repo.getLatest(citySlug);
    if (!latest) {
      return new Response(
        JSON.stringify({ error: "No observations yet for this city" }),
        { status: 404, headers: { "Content-Type": "application/json" } }
      );
    }

    // Get aggregates
    const hourlyStats = await repo.getHourlyStats(citySlug, hours);
    const dailyStats = await repo.getDailyStats(citySlug, Math.ceil(hours / 24));

    // Get trend
    const trend = await repo.trending(citySlug, hours);

    // Get AQI category
    const aqiCategory = getAQICategory(latest.us_aqi);
    const healthAdvisory = getHealthAdvisory(aqiCategory.name);

    // Filter metrics if requested
    const filterMetrics = (row) => {
      if (metric === "all") return row;
      const filtered = { hour: row.hour || row.day, obs_count: row.obs_count };
      const metricKey = metric.replace("_", "_").toLowerCase();
      Object.keys(row).forEach((key) => {
        if (key.includes(metricKey)) {
          filtered[key] = row[key];
        }
      });
      return filtered;
    };

    return new Response(
      JSON.stringify({
        city: {
          slug: city.slug,
          name: city.name,
          country: city.country,
          latitude: city.latitude,
          longitude: city.longitude,
        },
        latest: {
          observed_at: latest.observed_at,
          us_aqi: latest.us_aqi,
          pm2_5: latest.pm2_5,
          pm10: latest.pm10,
          carbon_monoxide: latest.carbon_monoxide,
          nitrogen_dioxide: latest.nitrogen_dioxide,
          sulphur_dioxide: latest.sulphur_dioxide,
          ozone: latest.ozone,
          ingested_at: latest.ingested_at,
        },
        hourly_stats: hourlyStats.map(filterMetrics),
        daily_stats: dailyStats.map(filterMetrics),
        trend,
        aqi_category: {
          value: aqiCategory.name,
          min: aqiCategory.min,
          max: aqiCategory.max,
          color: aqiCategory.color,
          emoji: aqiCategory.emoji,
        },
        health_advisory: healthAdvisory,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  } catch (error) {
    console.error("Error in GET /api/observations/[citySlug]:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}
