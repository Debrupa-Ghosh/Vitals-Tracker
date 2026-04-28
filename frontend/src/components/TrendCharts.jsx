import { useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
} from 'chart.js'
import { Line } from 'react-chartjs-2'

ChartJS.register(
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler
)

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: '#8a8880', font: { family: 'Inter', size: 12 } }
    },
    tooltip: {
      backgroundColor: '#1a1a2e',
      titleColor: '#e8e6df',
      bodyColor: '#e8e6df',
      borderColor: 'rgba(255,255,255,0.1)',
      borderWidth: 1,
      cornerRadius: 8,
      padding: 12,
    }
  },
  scales: {
    x: {
      grid: { color: 'rgba(255,255,255,0.04)' },
      ticks: { color: '#5a5850', font: { family: 'Inter', size: 11 } }
    },
    y: {
      grid: { color: 'rgba(255,255,255,0.04)' },
      ticks: { color: '#5a5850', font: { family: 'Inter', size: 11 } }
    }
  },
  elements: {
    line: { tension: 0.4 },
    point: { radius: 4, hoverRadius: 6 }
  },
  animation: { duration: 1200, easing: 'easeOutQuart' }
}

function makeGradient(ctx, color1, color2) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 300)
  gradient.addColorStop(0, color1)
  gradient.addColorStop(1, color2)
  return gradient
}

export default function TrendCharts({ trends }) {
  const data = trends?.trends || []
  const labels = data.map(d => d.label)

  const bpData = {
    labels,
    datasets: [
      {
        label: 'Systolic',
        data: data.map(d => d.systolic),
        borderColor: '#ff6b6b',
        backgroundColor: 'rgba(255,107,107,0.1)',
        fill: true,
      },
      {
        label: 'Diastolic',
        data: data.map(d => d.diastolic),
        borderColor: '#4f8cff',
        backgroundColor: 'rgba(79,140,255,0.1)',
        fill: true,
      },
    ],
  }

  const metricsData = {
    labels,
    datasets: [
      {
        label: 'BMI',
        data: data.map(d => d.bmi),
        borderColor: '#00d4aa',
        backgroundColor: 'rgba(0,212,170,0.1)',
        fill: true,
        yAxisID: 'y',
      },
      {
        label: 'Glucose',
        data: data.map(d => d.glucose),
        borderColor: '#ffb347',
        backgroundColor: 'rgba(255,179,71,0.1)',
        fill: true,
        yAxisID: 'y1',
      },
    ],
  }

  const metricsOptions = {
    ...commonOptions,
    scales: {
      ...commonOptions.scales,
      y: {
        ...commonOptions.scales.y,
        position: 'left',
        title: { display: true, text: 'BMI', color: '#5a5850' },
      },
      y1: {
        ...commonOptions.scales.y,
        position: 'right',
        title: { display: true, text: 'Glucose (mg/dL)', color: '#5a5850' },
        grid: { drawOnChartArea: false },
      },
    },
  }

  const healthScoreData = {
    labels,
    datasets: [
      {
        label: 'Health Score',
        data: data.map(d => d.health_score),
        borderColor: '#00d4aa',
        backgroundColor: 'rgba(0,212,170,0.15)',
        fill: true,
        borderWidth: 3,
      },
    ],
  }

  return (
    <div className="trend-charts">
      <h3 className="section-title">
        <span className="section-icon">📈</span>
        7-Day Health Trends
        {trends?.data_source === 'simulated' && (
          <span className="data-badge">Simulated Data</span>
        )}
      </h3>

      <div className="charts-grid">
        <div className="chart-card glass-card">
          <h4 className="chart-title">Blood Pressure</h4>
          <div className="chart-wrap">
            <Line data={bpData} options={commonOptions} />
          </div>
        </div>

        <div className="chart-card glass-card">
          <h4 className="chart-title">BMI & Glucose</h4>
          <div className="chart-wrap">
            <Line data={metricsData} options={metricsOptions} />
          </div>
        </div>

        <div className="chart-card glass-card chart-full">
          <h4 className="chart-title">Overall Health Score</h4>
          <div className="chart-wrap">
            <Line data={healthScoreData} options={commonOptions} />
          </div>
        </div>
      </div>
    </div>
  )
}
