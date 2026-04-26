const { Pool } = require("pg");

const pool = new Pool({
  host: process.env.AQ_DB_HOST || process.env.POSTGRES_HOST || "127.0.0.1",
  port: Number(process.env.AQ_DB_PORT || process.env.POSTGRES_PORT || 5432),
  database: process.env.AQ_DB_NAME || process.env.POSTGRES_DB || "postgres",
  user: process.env.AQ_DB_USER || process.env.POSTGRES_USER || "postgres",
  password: process.env.AQ_DB_PASSWORD || process.env.POSTGRES_PASSWORD || "postgres",
  max: Number(process.env.AQ_DB_POOL_MAX || 10),
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

pool.on("error", (error) => {
  console.error("Unexpected PostgreSQL pool error", error);
});

module.exports = pool;
