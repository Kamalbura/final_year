'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Navbar.module.css';

export default function Navbar() {
  const pathname = usePathname();
  const airflowUrl = process.env.NEXT_PUBLIC_AIRFLOW_URL || 'http://127.0.0.1:8080/home';

  const isActive = (path) => pathname === path || pathname.startsWith(path + '/');

  return (
    <nav className={styles.navbar}>
      <div className={styles.container}>
        <Link href="/" className={styles.brand}>
          <span className={styles.brandDot} aria-hidden="true"></span>
          <span className={styles.brandText}>Air Quality Platform</span>
        </Link>

        <div className={styles.links}>
          <Link
            href="/"
            className={`${styles.navLink} ${pathname === '/' ? styles.active : ''}`}
          >
            Overview
          </Link>

          <Link
            href="/air-quality"
            className={`${styles.navLink} ${isActive('/air-quality') ? styles.active : ''}`}
          >
            Air Quality
          </Link>

          <Link
            href="/dashboard/cities"
            className={`${styles.navLink} ${isActive('/dashboard/cities') ? styles.active : ''}`}
          >
            Legacy Route
          </Link>

          <a href={airflowUrl} target="_blank" rel="noopener noreferrer" className={styles.navLink}>
            Airflow DAGs
          </a>
        </div>
      </div>
    </nav>
  );
}
