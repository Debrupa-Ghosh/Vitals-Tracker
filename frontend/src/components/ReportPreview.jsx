import { useRef } from 'react'
import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'

export default function ReportPreview({ analysis, trends }) {
  const { demographics, vitals, lifestyle, bmi, bloodPressure, cholesterol, glucose, heartRisk, overallHealth } = analysis

  const generatePDF = () => {
    const doc = new jsPDF()
    const pageWidth = doc.internal.pageSize.getWidth()

    const buildPdf = (logoImg) => {
      // Header
      doc.setFillColor(15, 17, 23)
      doc.rect(0, 0, pageWidth, 40, 'F')
      
      let textX = 14
      if (logoImg) {
        doc.addImage(logoImg, 'PNG', 14, 10, 20, 20)
        textX = 40
      }

      doc.setTextColor(0, 212, 170)
      doc.setFontSize(22)
      doc.setFont('helvetica', 'bold')
      doc.text('Vitals Tracker', textX, 20)
      doc.setFontSize(11)
      doc.setTextColor(200, 200, 200)
      doc.text('Health Monitoring Report', textX, 30)
      doc.setFontSize(9)
      doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth - 14, 30, { align: 'right' })

      let y = 50

      // Patient Info
      doc.setTextColor(40, 40, 40)
      doc.setFontSize(14)
      doc.setFont('helvetica', 'bold')
      doc.text('Patient Information', 14, y)
      y += 6

      autoTable(doc, {
        startY: y,
        head: [['Field', 'Value']],
        body: [
          ['Name', demographics.name],
          ['Age', `${demographics.age} years`],
          ['Gender', demographics.gender],
          ['Height', `${vitals.height} cm`],
          ['Weight', `${vitals.weight} kg`],
        ],
        theme: 'striped',
        headStyles: { fillColor: [0, 212, 170], textColor: 255 },
        margin: { left: 14, right: 14 },
      })

      y = doc.lastAutoTable.finalY + 12

      // Vitals Analysis
      doc.setFontSize(14)
      doc.setFont('helvetica', 'bold')
      doc.text('Health Analysis', 14, y)
      y += 6

      autoTable(doc, {
        startY: y,
        head: [['Metric', 'Value', 'Category', 'Status']],
        body: [
          ['BMI', `${bmi.value} kg/m²`, bmi.category, bmi.severity],
          ['Blood Pressure', `${bloodPressure.systolic}/${bloodPressure.diastolic} mmHg`, bloodPressure.category, bloodPressure.severity],
          ['Cholesterol', `${cholesterol.value} mg/dL`, cholesterol.category, cholesterol.severity],
          ['Glucose', `${glucose.value} mg/dL`, glucose.category, glucose.severity],
        ],
        theme: 'striped',
        headStyles: { fillColor: [79, 140, 255], textColor: 255 },
        margin: { left: 14, right: 14 },
        didParseCell: function(data) {
          if (data.column.index === 3 && data.section === 'body') {
            const val = data.cell.raw
            if (val === 'healthy') data.cell.styles.textColor = [0, 180, 130]
            else if (val === 'moderate') data.cell.styles.textColor = [220, 160, 50]
            else if (val === 'critical') data.cell.styles.textColor = [220, 80, 80]
          }
        }
      })

      y = doc.lastAutoTable.finalY + 12

      // Heart Risk
      doc.setFontSize(14)
      doc.setFont('helvetica', 'bold')
      doc.text('Heart Disease Risk Assessment', 14, y)
      y += 8

      doc.setFontSize(11)
      doc.setFont('helvetica', 'normal')
      doc.text(`Risk Level: ${heartRisk.level}`, 14, y)
      y += 6
      doc.text(`Risk Score: ${(heartRisk.score * 100).toFixed(1)}%`, 14, y)
      y += 6
      doc.text(`Confidence: ${(heartRisk.confidence * 100).toFixed(0)}%`, 14, y)
      y += 6
      doc.text(`Contributing Factors: ${heartRisk.factors.join(', ')}`, 14, y, { maxWidth: pageWidth - 28 })
      y += 12

      // Overall Score
      doc.setFontSize(14)
      doc.setFont('helvetica', 'bold')
      doc.text('Overall Health Score', 14, y)
      y += 8
      doc.setFontSize(24)
      doc.setTextColor(0, 212, 170)
      doc.text(`${overallHealth.score}/100 (${overallHealth.grade})`, 14, y)
      y += 8
      doc.setFontSize(11)
      doc.setTextColor(100, 100, 100)
      doc.text(overallHealth.summary, 14, y, { maxWidth: pageWidth - 28 })

      // Lifestyle
      y += 14
      doc.setTextColor(40, 40, 40)
      doc.setFontSize(14)
      doc.setFont('helvetica', 'bold')
      doc.text('Lifestyle Factors', 14, y)
      y += 6

      autoTable(doc, {
        startY: y,
        head: [['Factor', 'Value']],
        body: [
          ['Smoking', lifestyle.smoking ? 'Yes' : 'No'],
          ['Alcohol', lifestyle.alcohol],
          ['Exercise', lifestyle.exercise],
          ['Diet', lifestyle.diet],
        ],
        theme: 'striped',
        headStyles: { fillColor: [255, 179, 71], textColor: 40 },
        margin: { left: 14, right: 14 },
      })

      // Footer
      const pageCount = doc.internal.getNumberOfPages()
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i)
        doc.setFontSize(8)
        doc.setTextColor(150)
        doc.text('Vitals Tracker — Multi-Agent Health Monitoring System', 14, doc.internal.pageSize.getHeight() - 10)
        doc.text(`Page ${i} of ${pageCount}`, pageWidth - 14, doc.internal.pageSize.getHeight() - 10, { align: 'right' })
      }

      doc.save(`vitals-report-${demographics.name.replace(/\s+/g, '-').toLowerCase()}.pdf`)
    }

    const img = new Image()
    img.src = '/logo.png'
    img.onload = () => buildPdf(img)
    img.onerror = () => buildPdf(null)
  }

  return (
    <div className="report-page">
      <div className="report-container">
        <div className="report-header">
          <h1 className="page-title">Health Report</h1>
          <button className="btn btn-primary btn-glow" onClick={generatePDF} id="download-report">
            📄 Download PDF Report
          </button>
        </div>

        <div className="report-preview glass-card">
          <div className="report-brand">
            <h2>Vitals Tracker Health Report</h2>
            <span className="report-date">{new Date().toLocaleDateString('en-US', { dateStyle: 'long' })}</span>
          </div>

          <div className="report-section">
            <h3>Patient: {demographics.name}</h3>
            <p>Age: {demographics.age} · Gender: {demographics.gender} · Height: {vitals.height}cm · Weight: {vitals.weight}kg</p>
          </div>

          <div className="report-grid">
            <div className="report-metric">
              <span className="report-metric-label">BMI</span>
              <span className={`report-metric-value severity-${bmi.severity}`}>{bmi.value}</span>
              <span className="report-metric-cat">{bmi.category}</span>
            </div>
            <div className="report-metric">
              <span className="report-metric-label">Blood Pressure</span>
              <span className={`report-metric-value severity-${bloodPressure.severity}`}>
                {bloodPressure.systolic}/{bloodPressure.diastolic}
              </span>
              <span className="report-metric-cat">{bloodPressure.category}</span>
            </div>
            <div className="report-metric">
              <span className="report-metric-label">Cholesterol</span>
              <span className={`report-metric-value severity-${cholesterol.severity}`}>{cholesterol.value}</span>
              <span className="report-metric-cat">{cholesterol.category}</span>
            </div>
            <div className="report-metric">
              <span className="report-metric-label">Glucose</span>
              <span className={`report-metric-value severity-${glucose.severity}`}>{glucose.value}</span>
              <span className="report-metric-cat">{glucose.category}</span>
            </div>
          </div>

          <div className="report-risk-section">
            <h3>Heart Disease Risk: <span className={`risk-level-${heartRisk.level.toLowerCase()}`}>{heartRisk.level}</span></h3>
            <p>Score: {(heartRisk.score * 100).toFixed(1)}% · Confidence: {(heartRisk.confidence * 100).toFixed(0)}%</p>
          </div>

          <div className="report-score-section">
            <span className="report-big-score">{overallHealth.score}</span>
            <span className="report-big-grade">{overallHealth.grade}</span>
            <p>{overallHealth.summary}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
