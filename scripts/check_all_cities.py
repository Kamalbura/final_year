import psycopg2
conn = psycopg2.connect(host='localhost', port=5432, database='airflow', user='airflow', password='airflow')
cur = conn.cursor()
for city in ['delhi', 'bengaluru', 'hyderabad']:
    cur.execute("SELECT COUNT(*) FROM aq.observations WHERE city_slug=%s", (city,))
    cnt = cur.fetchone()[0]
    cur.execute("SELECT MIN(observed_at), MAX(observed_at) FROM aq.observations WHERE city_slug=%s", (city,))
    mn, mx = cur.fetchone()
    print(f'{city}: {cnt} records, from {mn} to {mx}')
conn.close()
