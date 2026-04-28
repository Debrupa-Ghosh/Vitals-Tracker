import { useState } from 'react'

export default function Navbar({ tabs, activeTab, onTabChange }) {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <nav className="navbar" id="main-nav">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <img src="/logo.png" alt="Logo" className="brand-logo" style={{ height: '36px' }} />
          <span className="brand-text">Vitals-Tracker</span>
        </div>

        <button
          className="navbar-toggle"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle navigation"
        >
          <span></span><span></span><span></span>
        </button>

        <div className={`navbar-tabs ${mobileOpen ? 'open' : ''}`}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              id={`nav-tab-${tab.id}`}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''} ${tab.disabled ? 'disabled' : ''}`}
              onClick={() => {
                if (!tab.disabled) {
                  onTabChange(tab.id)
                  setMobileOpen(false)
                }
              }}
              disabled={tab.disabled}
            >
              <span className="nav-tab-icon">{tab.icon}</span>
              <span className="nav-tab-label">{tab.label}</span>
            </button>
          ))}
        </div>
      </div>
    </nav>
  )
}
