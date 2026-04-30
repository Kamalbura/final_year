const CITIES = [
  { slug: "delhi", name: "Delhi", country: "India", latitude: 28.6139, longitude: 77.209 },
  { slug: "mumbai", name: "Mumbai", country: "India", latitude: 19.076, longitude: 72.8777 },
  { slug: "bengaluru", name: "Bengaluru", country: "India", latitude: 12.9716, longitude: 77.5946 },
  { slug: "hyderabad", name: "Hyderabad", country: "India", latitude: 17.385, longitude: 78.4867 },
  { slug: "chennai", name: "Chennai", country: "India", latitude: 13.0827, longitude: 80.2707 },
  { slug: "kolkata", name: "Kolkata", country: "India", latitude: 22.5726, longitude: 88.3639 },
  { slug: "pune", name: "Pune", country: "India", latitude: 18.5204, longitude: 73.8567 },
  { slug: "ahmedabad", name: "Ahmedabad", country: "India", latitude: 23.0225, longitude: 72.5714 },
  { slug: "jaipur", name: "Jaipur", country: "India", latitude: 26.9124, longitude: 75.7873 },
  { slug: "lucknow", name: "Lucknow", country: "India", latitude: 26.8467, longitude: 80.9462 },
  { slug: "surat", name: "Surat", country: "India", latitude: 21.1702, longitude: 72.8311 },
  { slug: "kanpur", name: "Kanpur", country: "India", latitude: 26.4499, longitude: 80.3319 },
  { slug: "nagpur", name: "Nagpur", country: "India", latitude: 21.1458, longitude: 79.0882 },
  { slug: "bhopal", name: "Bhopal", country: "India", latitude: 23.2599, longitude: 77.4126 },
  { slug: "visakhapatnam", name: "Visakhapatnam", country: "India", latitude: 17.6868, longitude: 83.2185 },
  { slug: "new-york-city", name: "New York City", country: "United States", latitude: 40.7128, longitude: -74.006 },
  { slug: "los-angeles", name: "Los Angeles", country: "United States", latitude: 34.0522, longitude: -118.2437 },
  { slug: "chicago", name: "Chicago", country: "United States", latitude: 41.8781, longitude: -87.6298 },
  { slug: "houston", name: "Houston", country: "United States", latitude: 29.7604, longitude: -95.3698 },
  { slug: "san-francisco", name: "San Francisco", country: "United States", latitude: 37.7749, longitude: -122.4194 },
  { slug: "london", name: "London", country: "United Kingdom", latitude: 51.5072, longitude: -0.1276 },
  { slug: "manchester", name: "Manchester", country: "United Kingdom", latitude: 53.4808, longitude: -2.2426 },
  { slug: "birmingham", name: "Birmingham", country: "United Kingdom", latitude: 52.4862, longitude: -1.8904 },
  { slug: "toronto", name: "Toronto", country: "Canada", latitude: 43.6532, longitude: -79.3832 },
  { slug: "vancouver", name: "Vancouver", country: "Canada", latitude: 49.2827, longitude: -123.1207 },
  { slug: "paris", name: "Paris", country: "France", latitude: 48.8566, longitude: 2.3522 },
  { slug: "berlin", name: "Berlin", country: "Germany", latitude: 52.52, longitude: 13.405 },
  { slug: "amsterdam", name: "Amsterdam", country: "Netherlands", latitude: 52.3676, longitude: 4.9041 },
  { slug: "dubai", name: "Dubai", country: "United Arab Emirates", latitude: 25.2048, longitude: 55.2708 },
  { slug: "singapore", name: "Singapore", country: "Singapore", latitude: 1.3521, longitude: 103.8198 },
  { slug: "tokyo", name: "Tokyo", country: "Japan", latitude: 35.6762, longitude: 139.6503 },
  { slug: "seoul", name: "Seoul", country: "South Korea", latitude: 37.5665, longitude: 126.978 },
  { slug: "sydney", name: "Sydney", country: "Australia", latitude: -33.8688, longitude: 151.2093 },
  { slug: "melbourne", name: "Melbourne", country: "Australia", latitude: -37.8136, longitude: 144.9631 },
  { slug: "lagos", name: "Lagos", country: "Nigeria", latitude: 6.5244, longitude: 3.3792 },
  { slug: "cape-town", name: "Cape Town", country: "South Africa", latitude: -33.9249, longitude: 18.4241 },
];

function normalize(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function warehouseSlug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function enrichCity(city) {
  return {
    ...city,
    warehouseSlug: city.warehouseSlug || warehouseSlug(city.name),
  };
}

function getAllCities() {
  return CITIES.map(enrichCity);
}

function getSingleCity(slugOrName) {
  const target = normalize(slugOrName);
  const targetWarehouse = warehouseSlug(slugOrName);
  const city = CITIES.find((entry) => {
    const enriched = enrichCity(entry);
    return (
      entry.slug === target ||
      normalize(entry.name) === target ||
      enriched.warehouseSlug === targetWarehouse
    );
  });
  return city ? enrichCity(city) : null;
}

function getCityForWarehouseSlug(slug) {
  const targetWarehouse = warehouseSlug(slug);
  const city = CITIES.find((entry) => enrichCity(entry).warehouseSlug === targetWarehouse);
  return city ? enrichCity(city) : null;
}

module.exports = {
  CITIES,
  getAllCities,
  getSingleCity,
  getCityForWarehouseSlug,
};
