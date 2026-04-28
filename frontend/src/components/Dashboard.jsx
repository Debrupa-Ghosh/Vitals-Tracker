import MetricCard from './MetricCard.jsx'
import HealthGauge from './HealthGauge.jsx'
import TrendCharts from './TrendCharts.jsx'

export default function Dashboard({ analysis, trends, doctors }) {
  const { bmi, bloodPressure, cholesterol, glucose, heartRisk, overallHealth, demographics, lifestyle } = analysis

  return (
    <div className="dashboard-page">
      <div className="dashboard-header">
        <h1 className="page-title">Health Dashboard</h1>
        <p className="page-subtitle">
          Analysis for <strong>{demographics.name}</strong> · {demographics.age} years · {demographics.gender}
        </p>
      </div>

      <div className="dashboard-grid">
        {/* Health Gauge */}
        <div className="dashboard-gauge">
          <HealthGauge
            score={overallHealth.score}
            grade={overallHealth.grade}
            summary={overallHealth.summary}
          />
        </div>

        {/* Metric Cards */}
        <div className="dashboard-metrics">
          <MetricCard
            title="Body Mass Index"
            value={bmi.value}
            unit="kg/m²"
            category={bmi.category}
            severity={bmi.severity}
            icon="⚖️"
          />
          <MetricCard
            title="Blood Pressure"
            value={`${bloodPressure.systolic}/${bloodPressure.diastolic}`}
            unit="mmHg"
            category={bloodPressure.category}
            severity={bloodPressure.severity}
            icon="💓"
          />
          <MetricCard
            title="Cholesterol"
            value={cholesterol.value}
            unit="mg/dL"
            category={cholesterol.category}
            severity={cholesterol.severity}
            icon="🩸"
          />
          <MetricCard
            title="Fasting Glucose"
            value={glucose.value}
            unit="mg/dL"
            category={glucose.category}
            severity={glucose.severity}
            icon="🍬"
          />
        </div>

        {/* Heart Risk Banner */}
        <div className={`risk-banner glass-card risk-${heartRisk.level.toLowerCase()}`}>
          <div className="risk-header">
            <span className="risk-icon">
              {heartRisk.level === 'Low' ? '💚' : heartRisk.level === 'Medium' ? '💛' : '❤️‍🔥'}
            </span>
            <div>
              <h3 className="risk-title">Heart Disease Risk: {heartRisk.level}</h3>
              <p className="risk-detail">
                Risk Score: {(heartRisk.score * 100).toFixed(1)}% · Confidence: {(heartRisk.confidence * 100).toFixed(0)}%
              </p>
            </div>
          </div>
          {heartRisk.factors.length > 0 && (
            <div className="risk-factors">
              <span className="risk-factors-label">Contributing factors:</span>
              <div className="risk-tags">
                {heartRisk.factors.map((f, i) => (
                  <span key={i} className="risk-tag">{f}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Lifestyle & Habits */}
        <div className="lifestyle-section glass-card" style={{ gridColumn: '1 / -1', padding: '24px' }}>
          <h3 className="section-title">
            <span className="section-icon">🏃</span>
            Lifestyle & Habits
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px', marginTop: '16px' }}>
            <div className="lifestyle-item">
              <span className="form-label">Smoking</span>
              <div style={{ color: lifestyle.smoking ? 'var(--danger)' : 'var(--healthy)', fontWeight: '600', fontSize: '1.1rem' }}>
                {lifestyle.smoking ? '🚬 Smoker' : '✅ Non-smoker'}
              </div>
            </div>
            <div className="lifestyle-item">
              <span className="form-label">Alcohol</span>
              <div style={{ color: 'var(--text)', fontWeight: '600', fontSize: '1.1rem', textTransform: 'capitalize' }}>
                🍷 {lifestyle.alcohol}
              </div>
            </div>
            <div className="lifestyle-item">
              <span className="form-label">Exercise</span>
              <div style={{ color: 'var(--text)', fontWeight: '600', fontSize: '1.1rem', textTransform: 'capitalize' }}>
                🏃 {lifestyle.exercise}
              </div>
            </div>
            <div className="lifestyle-item">
              <span className="form-label">Diet</span>
              <div style={{ color: 'var(--text)', fontWeight: '600', fontSize: '1.1rem', textTransform: 'capitalize' }}>
                🥗 {lifestyle.diet}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Trend Charts */}
      <TrendCharts trends={trends} />

      {/* Suggested Doctors */}
      {doctors && doctors.length > 0 && (
        <div className="doctors-section" style={{ marginTop: '24px' }}>
          <h3 className="section-title">
            <span className="section-icon">👨‍⚕️</span>
            Suggested Doctors
          </h3>
          <div className="doctors-list" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
            {doctors.map((doc, i) => (
              <div key={i} className="doctor-card glass-card">
                <div className="doctor-header">
                  <div className="doctor-avatar">
                    {doc.name.split(' ').map(w => w[0]).join('').slice(0, 2)}
                  </div>
                  <div>
                    <h4 className="doctor-name">{doc.name}</h4>
                    <span className="doctor-specialty">{doc.specialty}</span>
                  </div>
                </div>
                <div className="doctor-details">
                  <span>🏨 {doc.hospital}</span>
                  <span>📍 {doc.distance}</span>
                  <span>⭐ {doc.rating}</span>
                  <span>📞 {doc.phone}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
