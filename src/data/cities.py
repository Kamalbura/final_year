from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float

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


def city_by_slug() -> dict[str, City]:
    return {city.slug: city for city in INDIA_MAJOR_CITIES}