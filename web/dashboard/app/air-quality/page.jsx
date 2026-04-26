import CitiesGrid from '../../components/CitiesGrid';

export const metadata = {
  title: 'Live Air Quality Dashboard',
  description: 'Live AQI ranking and pollutant metrics across all cities',
};

export default function AirQualityPage() {
  return <CitiesGrid />;
}
