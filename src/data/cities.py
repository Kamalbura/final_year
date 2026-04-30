from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    country: str = "India"
    country_code: str = "IN"
    aqi_standard: str = "US AQI"

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")


INDIA_MAJOR_CITIES: tuple[City, ...] = (
    City("Delhi", 28.6139, 77.2090),
    City("Mumbai", 19.0760, 72.8777),
    City("Bengaluru", 12.9716, 77.5946),
    City("Hyderabad", 17.3850, 78.4867),
    City("Chennai", 13.0827, 80.2707),
    City("Kolkata", 22.5726, 88.3639),
    City("Pune", 18.5204, 73.8567),
    City("Ahmedabad", 23.0225, 72.5714),
    City("Jaipur", 26.9124, 75.7873),
    City("Lucknow", 26.8467, 80.9462),
    City("Surat", 21.1702, 72.8311),
    City("Kanpur", 26.4499, 80.3319),
    City("Nagpur", 21.1458, 79.0882),
    City("Bhopal", 23.2599, 77.4126),
    City("Visakhapatnam", 17.6868, 83.2185),
)


GLOBAL_MAJOR_CITIES: tuple[City, ...] = (
    City("New York City", 40.7128, -74.0060, country="United States", country_code="US"),
    City("Los Angeles", 34.0522, -118.2437, country="United States", country_code="US"),
    City("Chicago", 41.8781, -87.6298, country="United States", country_code="US"),
    City("Houston", 29.7604, -95.3698, country="United States", country_code="US"),
    City("San Francisco", 37.7749, -122.4194, country="United States", country_code="US"),
    City("London", 51.5074, -0.1278, country="United Kingdom", country_code="GB"),
    City("Manchester", 53.4808, -2.2426, country="United Kingdom", country_code="GB"),
    City("Birmingham", 52.4862, -1.8904, country="United Kingdom", country_code="GB"),
    City("Toronto", 43.6532, -79.3832, country="Canada", country_code="CA"),
    City("Vancouver", 49.2827, -123.1207, country="Canada", country_code="CA"),
    City("Paris", 48.8566, 2.3522, country="France", country_code="FR"),
    City("Berlin", 52.5200, 13.4050, country="Germany", country_code="DE"),
    City("Amsterdam", 52.3676, 4.9041, country="Netherlands", country_code="NL"),
    City("Dubai", 25.2048, 55.2708, country="United Arab Emirates", country_code="AE"),
    City("Singapore", 1.3521, 103.8198, country="Singapore", country_code="SG"),
    City("Tokyo", 35.6762, 139.6503, country="Japan", country_code="JP"),
    City("Seoul", 37.5665, 126.9780, country="South Korea", country_code="KR"),
    City("Sydney", -33.8688, 151.2093, country="Australia", country_code="AU"),
    City("Melbourne", -37.8136, 144.9631, country="Australia", country_code="AU"),
    City("Lagos", 6.5244, 3.3792, country="Nigeria", country_code="NG"),
    City("Cape Town", -33.9249, 18.4241, country="South Africa", country_code="ZA"),
)

ALL_MAJOR_CITIES: tuple[City, ...] = INDIA_MAJOR_CITIES + GLOBAL_MAJOR_CITIES


def city_by_slug() -> dict[str, City]:
    cities: dict[str, City] = {}
    for city in ALL_MAJOR_CITIES:
        cities[city.slug] = city
        cities[city.slug.replace("_", "-")] = city
    return cities


def city_catalog() -> tuple[City, ...]:
    return ALL_MAJOR_CITIES


def dag_id_for_city(city: City) -> str:
    return f"aq_{city.slug}_incremental_hourly"
