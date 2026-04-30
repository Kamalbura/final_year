# Service Links & Access Points

## Main Services

### Dashboard (Next.js Frontend)
- **URL**: http://100.111.13.58:3000/
- **Port**: 3000
- **Status**: Running
- **Components**: 
  - Homepage: `/`
  - Air Quality Dashboard: `/air-quality`
  - City Details: `/air-quality/city/[citySlug]`

### Airflow Web UI
- **URL**: http://100.111.13.58:8080/
- **Port**: 8080
- **Status**: Running
- **Functions**:
  - DAG Management
  - Task Monitoring
  - Scheduler Status
  - Logs & Troubleshooting

---

## API Endpoints (Backend)

### Dashboard API Routes
- **All Cities Observations**: http://100.111.13.58:3000/api/observations
- **Single City Observations**: http://100.111.13.58:3000/api/observations/[citySlug]

### Example City Slugs (Use in `/api/observations/[citySlug]`)
- `ahmedabad`
- `bengaluru`
- `delhi`
- `hyderabad`
- `jaipur`
- `kolkata`
- `mumbai`
- `pune`
- `visakhapatnam`
- `surat`
- `lucknow`
- `indore`
- `nagpur`
- `varanasi`
- `bhopal`

---

## Database & Storage

### PostgreSQL Database
- **Host**: localhost (in Docker network)
- **Port**: 5432
- **Database**: airflow
- **Schema**: `aq` (Air Quality data)
- **Status**: Healthy ✓
- **Access**: Via Docker Compose

### Data Storage
- **Location**: `/home/bura/projects/final_year/data/india_aq_1y/`
- **Files**: 15 city CSV archives
- **Format**: CSV with timestamps, measurements, metadata

---

## Monitoring & Administration

### SSH Access
```bash
ssh bura@100.111.13.58
```

### Directory Structure
```
~/projects/final_year/
├── web/dashboard/          # Next.js app (port 3000)
├── deployment/pi_airflow/  # Airflow configuration
├── data/india_aq_1y/       # CSV archives
└── src/                     # Python modules
```

### Docker Services (Airflow Stack)
```bash
cd ~/projects/pi_airflow
docker compose ps
```

- **airflow-webserver**: 0.0.0.0:8080
- **airflow-scheduler**: Running
- **airflow-triggerer**: Running
- **airflow-init**: Initialized
- **postgres**: 5432 (healthy)

---

## Quick Reference Links

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard Home | http://100.111.13.58:3000/ | Air quality UI |
| Cities List | http://100.111.13.58:3000/air-quality | Browse all cities |
| Airflow UI | http://100.111.13.58:8080/ | Task orchestration |
| Airflow DAGs | http://100.111.13.58:8080/dags | DAG management |
| Airflow Logs | http://100.111.13.58:8080/logs | Task logs |
| API (All) | http://100.111.13.58:3000/api/observations | All cities data |

---

## Troubleshooting Commands

### Check Dashboard Status
```bash
ssh bura@100.111.13.58 "ps aux | grep 'npm start'"
```

### Check Airflow Services
```bash
ssh bura@100.111.13.58 "cd ~/projects/pi_airflow && docker compose ps"
```

### View Dashboard Logs
```bash
ssh bura@100.111.13.58 "tail -100 ~/projects/final_year/web/dashboard/dashboard.log"
```

### Check PostgreSQL Health
```bash
ssh bura@100.111.13.58 "docker compose -f ~/projects/pi_airflow/docker-compose.yml ps postgres"
```

---

**Last Updated**: 2026-04-26
**RPi IP**: 100.111.13.58
**All services**: ✅ Operational & Verified
