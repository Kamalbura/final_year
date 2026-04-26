import CityDashboard from '../../../../components/CityDashboard';

export const metadata = {
  title: 'City Air Quality Details',
  description: 'Live city-level AQI trends and health advisory',
};

export default function AirQualityCityPage({ params }) {
  return <CityDashboard citySlug={params.citySlug} />;
}
