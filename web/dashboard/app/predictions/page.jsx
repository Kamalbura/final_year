import PredictionDashboard from '../../components/PredictionDashboard';

export const metadata = {
  title: 'AI Air Quality Predictions',
  description: '7-day air quality forecasts powered by deep learning models',
};

export default function PredictionsPage() {
  return <PredictionDashboard />;
}
