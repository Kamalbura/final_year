import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, database='airflow', user='airflow', password='airflow')
cur = conn.cursor()
cur.execute("SELECT observed_at, pm2_5, pm10, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, us_aqi FROM aq.observations WHERE city_slug=%s ORDER BY observed_at DESC LIMIT 5", ('hyderabad',))
for row in cur.fetchall():
    print(row)
print("---row count---")
cur.execute("SELECT COUNT(*) FROM aq.observations WHERE city_slug=%s", ('hyderabad',))
print(cur.fetchone())
conn.close()
