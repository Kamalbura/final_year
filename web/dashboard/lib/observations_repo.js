/**
 * Query observations from PostgreSQL with caching.
 * Exported for use in Next.js API routes.
 */

const pool = require("./db-pool");

class ObservationRepository {
  /**
   * Get latest observation for a city.
   */
  async getLatest(citySlug) {
    const result = await pool.query(
      `
      SELECT * FROM aq.latest_observations
      WHERE city_slug = $1
      LIMIT 1
      `,
      [citySlug]
    );
    return result.rows[0] || null;
  }

  /**
   * Get hourly aggregates for last N hours.
   */
  async getHourlyStats(citySlug, hours = 24) {
    const result = await pool.query(
      `
      SELECT 
        hour,
        obs_count,
        pm2_5_avg, pm2_5_max, pm2_5_min,
        pm10_avg, pm10_max, pm10_min,
        co_avg, co_max, co_min,
        no2_avg, no2_max, no2_min,
        so2_avg, so2_max, so2_min,
        o3_avg, o3_max, o3_min,
        us_aqi_avg, us_aqi_max, us_aqi_min
      FROM aq.hourly_aggregates
      WHERE city_slug = $1
        AND hour >= NOW() - INTERVAL '1 hour' * $2
      ORDER BY hour DESC
      `,
      [citySlug, hours]
    );
    return result.rows;
  }

  /**
   * Get daily aggregates for last N days.
   */
  async getDailyStats(citySlug, days = 30) {
    const result = await pool.query(
      `
      SELECT 
        day,
        obs_count,
        pm2_5_avg, pm2_5_max, pm2_5_min,
        pm10_avg, pm10_max, pm10_min,
        co_avg, co_max, co_min,
        no2_avg, no2_max, no2_min,
        so2_avg, so2_max, so2_min,
        o3_avg, o3_max, o3_min,
        us_aqi_avg, us_aqi_max, us_aqi_min
      FROM aq.daily_aggregates
      WHERE city_slug = $1
        AND day >= CURRENT_DATE - INTERVAL '1 day' * $2
      ORDER BY day DESC
      `,
      [citySlug, days]
    );
    return result.rows;
  }

  /**
   * Get latest observation for ALL cities (for ranking).
   */
  async allCitiesLatest() {
    const result = await pool.query(
      `
      SELECT 
        city_slug,
        city_name,
        observed_at,
        us_aqi,
        pm2_5,
        pm10,
        carbon_monoxide,
        nitrogen_dioxide,
        sulphur_dioxide,
        ozone
      FROM aq.latest_observations
      ORDER BY us_aqi DESC
      `
    );
    return result.rows;
  }

  /**
   * Calculate trend direction and slope.
   */
  async trending(citySlug, hours = 24) {
    const hourly = await this.getHourlyStats(citySlug, hours);
    
    if (hourly.length < 2) {
      return { direction: "stable", slope: 0, change_percent: 0 };
    }

    const last = hourly[0]; // Most recent
    const first = hourly[hourly.length - 1]; // Oldest

    const lastVal = last.us_aqi_avg || 0;
    const firstVal = first.us_aqi_avg || 0;

    const change = lastVal - firstVal;
    const changePercent =
      firstVal !== 0 ? (change / firstVal) * 100 : 0;
    const slope = change / hourly.length;

    let direction;
    if (Math.abs(changePercent) < 2) {
      direction = "stable";
    } else if (change > 0) {
      direction = "increasing";
    } else {
      direction = "decreasing";
    }

    return {
      direction,
      slope: Math.round(slope * 100) / 100,
      change_percent: Math.round(changePercent * 10) / 10,
      change_absolute: Math.round(change * 10) / 10,
    };
  }
}

module.exports = {
  ObservationRepository,
  getRepository: () => new ObservationRepository(),
};
