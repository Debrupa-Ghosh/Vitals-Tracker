export default function MetricCard({ title, value, unit, category, severity, icon }) {
  const severityClass = severity === 'healthy' ? 'metric-healthy'
    : severity === 'moderate' ? 'metric-warning'
    : 'metric-danger'

  return (
    <div className={`metric-card glass-card ${severityClass}`}>
      <div className="metric-header">
        <span className="metric-icon">{icon}</span>
        <span className={`metric-badge badge-${severity}`}>{category}</span>
      </div>
      <div className="metric-value-wrap">
        <span className="metric-value">{value}</span>
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
      <div className="metric-title">{title}</div>
    </div>
  )
}
