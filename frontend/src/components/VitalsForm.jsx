import { useState } from 'react'

const FALLBACK_SCHEMA = {
  steps: [
    {
      id: 'personal', title: 'Personal Info', icon: '👤',
      fields: [
        { name: 'name', type: 'text', label: 'Full Name', placeholder: 'Enter your full name', required: true },
        { name: 'age', type: 'number', label: 'Age', placeholder: 'e.g. 35', min: 1, max: 120, required: true, unit: 'years' },
        { name: 'gender', type: 'select', label: 'Gender', options: [{ value: 'male', label: 'Male' }, { value: 'female', label: 'Female' }, { value: 'other', label: 'Other' }], required: true },
      ]
    },
    {
      id: 'vitals', title: 'Vitals', icon: '❤️',
      fields: [
        { name: 'height', type: 'number', label: 'Height', placeholder: 'e.g. 175', min: 50, max: 300, required: true, unit: 'cm', step: 0.1 },
        { name: 'weight', type: 'number', label: 'Weight', placeholder: 'e.g. 72', min: 10, max: 500, required: true, unit: 'kg', step: 0.1 },
        { name: 'systolic', type: 'number', label: 'Systolic BP', placeholder: 'e.g. 120', min: 60, max: 250, required: true, unit: 'mmHg' },
        { name: 'diastolic', type: 'number', label: 'Diastolic BP', placeholder: 'e.g. 80', min: 40, max: 150, required: true, unit: 'mmHg' },
        { name: 'cholesterol', type: 'number', label: 'Total Cholesterol', placeholder: 'e.g. 200', min: 50, max: 500, required: true, unit: 'mg/dL' },
        { name: 'glucose', type: 'number', label: 'Fasting Glucose', placeholder: 'e.g. 95', min: 30, max: 600, required: true, unit: 'mg/dL' },
      ]
    },
    {
      id: 'lifestyle', title: 'Lifestyle', icon: '🏃',
      fields: [
        { name: 'smoking', type: 'toggle', label: 'Do you smoke?', required: false },
        { name: 'alcohol', type: 'select', label: 'Alcohol Consumption', options: [{ value: 'none', label: 'None' }, { value: 'light', label: 'Light' }, { value: 'moderate', label: 'Moderate' }, { value: 'heavy', label: 'Heavy' }], required: true },
        { name: 'exercise', type: 'select', label: 'Exercise Level', options: [{ value: 'sedentary', label: 'Sedentary' }, { value: 'light', label: 'Light' }, { value: 'moderate', label: 'Moderate' }, { value: 'active', label: 'Active' }], required: true },
        { name: 'diet', type: 'select', label: 'Diet Quality', options: [{ value: 'poor', label: 'Poor' }, { value: 'average', label: 'Average' }, { value: 'balanced', label: 'Balanced' }, { value: 'excellent', label: 'Excellent' }], required: true },
      ]
    }
  ]
}

export default function VitalsForm({ schema, onSubmit, loading, error }) {
  const formSchema = schema?.steps?.length ? schema : FALLBACK_SCHEMA
  const [step, setStep] = useState(0)
  const [formData, setFormData] = useState({
    smoking: false,
    alcohol: '',
    exercise: '',
    diet: '',
  })
  const [fieldErrors, setFieldErrors] = useState({})

  const currentStep = formSchema.steps[step]
  const isLast = step === formSchema.steps.length - 1

  const updateField = (name, value) => {
    setFormData(prev => ({ ...prev, [name]: value }))
    setFieldErrors(prev => ({ ...prev, [name]: null }))
  }

  const validateStep = () => {
    const errors = {}
    for (const field of currentStep.fields) {
      const val = formData[field.name]
      if (field.required && (val === undefined || val === '' || val === null)) {
        errors[field.name] = `${field.label} is required`
      }
      if (field.type === 'number' && val !== undefined && val !== '') {
        const num = Number(val)
        if (isNaN(num)) errors[field.name] = 'Must be a number'
        else if (field.min !== undefined && num < field.min) errors[field.name] = `Min: ${field.min}`
        else if (field.max !== undefined && num > field.max) errors[field.name] = `Max: ${field.max}`
      }
    }
    setFieldErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleNext = () => {
    if (validateStep()) setStep(s => s + 1)
  }

  const handleSubmit = (e) => {
    if (e && e.preventDefault) e.preventDefault()
    if (!isLast) {
      handleNext()
    } else {
      if (validateStep()) onSubmit(formData)
    }
  }

  const renderField = (field) => {
    const err = fieldErrors[field.name]

    if (field.type === 'toggle') {
      return (
        <div key={field.name} className="form-field" id={`field-${field.name}`}>
          <label className="form-label">{field.label}</label>
          <button
            type="button"
            className={`toggle-btn ${formData[field.name] ? 'active' : ''}`}
            onClick={() => updateField(field.name, !formData[field.name])}
          >
            <span className="toggle-track">
              <span className="toggle-thumb"></span>
            </span>
            <span className="toggle-label">{formData[field.name] ? 'Yes' : 'No'}</span>
          </button>
        </div>
      )
    }

    if (field.type === 'select') {
      return (
        <div key={field.name} className={`form-field ${err ? 'has-error' : ''}`} id={`field-${field.name}`}>
          <label className="form-label">{field.label}</label>
          <div className="select-wrap">
            <select
              className="form-select"
              value={formData[field.name] || ''}
              onChange={e => updateField(field.name, e.target.value)}
            >
              <option value="">Select...</option>
              {field.options.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          {err && <span className="field-error">{err}</span>}
        </div>
      )
    }

    return (
      <div key={field.name} className={`form-field ${err ? 'has-error' : ''}`} id={`field-${field.name}`}>
        <label className="form-label">{field.label}</label>
        <div className="input-wrap">
          <input
            type={field.type}
            className="form-input"
            placeholder={field.placeholder || ''}
            value={formData[field.name] || ''}
            onChange={e => updateField(field.name, e.target.value)}
            min={field.min}
            max={field.max}
            step={field.step}
          />
          {field.unit && <span className="input-unit">{field.unit}</span>}
        </div>
        {err && <span className="field-error">{err}</span>}
      </div>
    )
  }

  return (
    <div className="form-page">
      <div className="form-container glass-card">
        <div className="form-header">
          <h1 className="form-title">Health Assessment</h1>
          <p className="form-subtitle">Enter your vitals for a comprehensive AI-powered analysis</p>
        </div>

        {/* Progress */}
        <div className="step-progress">
          {formSchema.steps.map((s, i) => (
            <div key={s.id} className={`step-item ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}>
              <div className="step-dot">
                {i < step ? '✓' : i + 1}
              </div>
              <span className="step-label">{s.title}</span>
            </div>
          ))}
        </div>

        <div className="form-body">
          <div className="step-content">
            <h2 className="step-title">
              <span className="step-icon">
                {{'user': '👤', 'heart': '❤️', 'activity': '🏃'}[currentStep.icon] || currentStep.icon || '📋'}
              </span>
              {currentStep.title}
            </h2>
            <div className="fields-grid">
              {currentStep.fields.map(renderField)}
            </div>
          </div>

          {error && (
            <div className="form-error-banner">
              <span>⚠️</span> {error}
            </div>
          )}

          <div className="form-actions">
            {step > 0 && (
              <button type="button" className="btn btn-ghost" onClick={() => setStep(s => s - 1)}>
                ← Back
              </button>
            )}
            {!isLast ? (
              <button type="button" className="btn btn-primary" onClick={handleNext}>
                Next →
              </button>
            ) : (
              <button type="button" className="btn btn-primary btn-glow" onClick={handleSubmit} disabled={loading}>
                {loading ? (
                  <><span className="spinner"></span> Analyzing...</>
                ) : (
                  '🔬 Analyze My Health'
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
