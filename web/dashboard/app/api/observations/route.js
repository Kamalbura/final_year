/**
 * API Route: GET /api/observations
 * 
 * Returns latest observations for ALL cities with ranking.
 * 
 * Response: [
 *   {
 *     rank: 1,
 *     city: { slug, name, country },
 *     current_aqi: 142,
 *     category: "unhealthy",
 *     pm2_5: 42,
 *     pm10: 95,
 *     observed_at: "2026-04-26T14:00Z",
 *     color: "#7c2d12",
 *     emoji: "🟤"
 *   },
 *   ...
 * ]
 */

import { ObservationRepository } from "../../lib/observations_repo.js";
import { getCityForWarehouseSlug } from "../../lib/cities.js";

const AQI_CATEGORIES = {
  0: { name: "good", color: "#10b981", emoji: "✅" },
  50: { name: "moderate", color: "#f59e0b", emoji: "🟡" },
  100: { name: "unhealthy_sg", color: "#f97316", emoji: "🟠" },
  150: { name: "unhealthy", color: "#ef4444", emoji: "🔴" },
  200: { name: "very_unhealthy", color: "#9333ea", emoji: "🟣" },
  300: { name: "hazardous", color: "#7c2d12", emoji: "🟤" },
};

function getCategoryForAQI(aqi) {
  if (aqi === null || aqi === undefined)
    return AQI_CATEGORIES[0]; // good

  const thresholds = Object.keys(AQI_CATEGORIES)
    .map(Number)
    .sort((a, b) => b - a);
  for (const threshold of thresholds) {
    if (aqi >= threshold) {
      return AQI_CATEGORIES[threshold];
    }
  }
  return AQI_CATEGORIES[0];
}

export async function GET() {
  try {
    const repo = new ObservationRepository();

    // Get latest for all cities
    const allLatest = await repo.allCitiesLatest();

    // Rank by AQI (highest = worst)
    const ranked = allLatest
      .map((obs, index) => {
        const cityInfo = getCityForWarehouseSlug(obs.city_slug) || {};
        const category = getCategoryForAQI(obs.us_aqi);

        return {
          rank: index + 1,
          city: {
            slug: cityInfo.slug || obs.city_slug,
            warehouse_slug: obs.city_slug,
            name: obs.city_name,
            country: cityInfo.country || "Unknown",
          },
          current_aqi: obs.us_aqi,
          pm2_5: obs.pm2_5,
          pm10: obs.pm10,
          co: obs.carbon_monoxide,
          no2: obs.nitrogen_dioxide,
          so2: obs.sulphur_dioxide,
          o3: obs.ozone,
          category: category.name,
          color: category.color,
          emoji: category.emoji,
          observed_at: obs.observed_at,
        };
      })
      .sort((a, b) => (b.current_aqi || 0) - (a.current_aqi || 0))
      .map((item, idx) => ({ ...item, rank: idx + 1 }));

    return new Response(JSON.stringify(ranked), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error in GET /api/observations:", error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
