import { useEffect, useState } from 'react'

export default function HealthGauge({ score, grade, summary }) {
  const [animatedScore, setAnimatedScore] = useState(0)
  const radius = 80
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (animatedScore / 100) * circumference

  const getColor = (s) => {
    if (s >= 80) return '#00d4aa'
    if (s >= 60) return '#ffb347'
    return '#ff6b6b'
  }

  useEffect(() => {
    let frame
    const start = performance.now()
    const duration = 1500
    const animate = (now) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setAnimatedScore(Math.round(eased * score))
      if (progress < 1) frame = requestAnimationFrame(animate)
    }
    frame = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(frame)
  }, [score])

  return (
    <div className="health-gauge glass-card">
      <h3 className="gauge-title">Overall Health Score</h3>
      <div className="gauge-ring-wrap">
        <svg viewBox="0 0 200 200" className="gauge-svg">
          <circle
            cx="100" cy="100" r={radius}
            fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12"
          />
          <circle
            cx="100" cy="100" r={radius}
            fill="none" stroke={getColor(animatedScore)} strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            transform="rotate(-90 100 100)"
            style={{ transition: 'stroke 0.5s ease' }}
          />
        </svg>
        <div className="gauge-center">
          <span className="gauge-score" style={{ color: getColor(animatedScore) }}>
            {animatedScore}
          </span>
          <span className="gauge-grade">{grade}</span>
        </div>
      </div>
      <p className="gauge-summary">{summary}</p>
    </div>
  )
}
