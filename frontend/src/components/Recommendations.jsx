export default function Recommendations({ items }) {
  if (!items || items.length === 0) return null

  const priorityColors = {
    high: 'rec-high',
    medium: 'rec-medium',
    low: 'rec-low',
  }

  return (
    <div className="recommendations-section">
      <h3 className="section-title">
        <span className="section-icon">💡</span>
        Health Recommendations
      </h3>
      <div className="rec-list">
        {items.map((rec, i) => (
          <div key={i} className={`rec-card glass-card ${priorityColors[rec.priority] || ''}`}>
            <div className="rec-header">
              <span className="rec-icon">{rec.icon}</span>
              <span className="rec-category">{rec.category}</span>
              <span className={`rec-priority badge-${rec.priority === 'high' ? 'danger' : rec.priority === 'medium' ? 'warning' : 'healthy'}`}>
                {rec.priority}
              </span>
            </div>
            <p className="rec-text">{rec.text}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
