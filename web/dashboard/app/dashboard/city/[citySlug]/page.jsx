import CityDashboard from '../../../../components/CityDashboard';

export const metadata = {
  title: 'City Air Quality - Air Quality Dashboard',
  description: 'Detailed air quality information for a city',
};

export default function CityDetailPage({ params }) {
  return <CityDashboard citySlug={params.citySlug} />;
}
