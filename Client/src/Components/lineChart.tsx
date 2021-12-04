import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface LineChartProps {
  dataPointsSet: number[][];
  labels: any[];
  datasetNameSet: string[];
  borderColorSet: string[];
  backgroundColorSet: string[];
  title: string;
}

const LineChart = ({
  dataPointsSet,
  labels,
  datasetNameSet,
  borderColorSet,
  backgroundColorSet,
  title
}: LineChartProps) => {
  const datasets = dataPointsSet.map((datapoints, i) => {
    return {
      data: datapoints,
      label: datasetNameSet[i],
      borderColor: borderColorSet[i],
      backgroundColor: backgroundColorSet[i]
    };
  });
  const data = { labels, datasets };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const
      },
      title: {
        display: true,
        text: title
      }
    }
  };
  return <Line options={options} data={data} />;
};

export default LineChart;
