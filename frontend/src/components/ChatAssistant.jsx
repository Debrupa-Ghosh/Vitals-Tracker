import { useState, useRef, useEffect } from 'react'
import { sendChat } from '../api/vitalsApi.js'

export default function ChatAssistant({ userId }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hello! 👋 I'm your health assistant. You can ask me about your BMI, blood pressure, cholesterol, glucose levels, heart risk, or any health topic. How can I help?",
    }
  ])
  const [input, setInput] = useState('')
  const [typing, setTyping] = useState(false)
  const bottomRef = useRef(null)

  const quickReplies = [
    "What's my BMI?",
    "Is my blood pressure okay?",
    "Heart risk explained",
    "Exercise tips",
    "Find a doctor",
  ]

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, typing])

  const handleSend = async (text) => {
    const msg = text || input.trim()
    if (!msg) return

    setMessages(prev => [...prev, { role: 'user', text: msg }])
    setInput('')
    setTyping(true)

    try {
      const res = await sendChat(msg, userId)
      setMessages(prev => [...prev, { role: 'assistant', text: res.response }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: "Sorry, I couldn't process your request. Please make sure the backend is running."
      }])
    }
    setTyping(false)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-container glass-card" id="chat-assistant">
      <div className="chat-header">
        <div className="chat-avatar-wrap">
          <div className="chat-bot-avatar">🤖</div>
          <span className="chat-online-dot"></span>
        </div>
        <div>
          <h3 className="chat-title">Health Assistant</h3>
          <span className="chat-status">HealthBot · Always Available</span>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble-wrapper ${msg.role}`}>
            {msg.role === 'assistant' && <div className="chat-avatar bot-avatar">🤖</div>}
            <div className={`chat-bubble ${msg.role}`}>
              <div className="bubble-content">
                {msg.text.split('\n').map((line, j) => (
                  <p key={j}>{line.replace(/\*\*(.*?)\*\*/g, (_, t) => t)}</p>
                ))}
              </div>
            </div>
            {msg.role === 'user' && <div className="chat-avatar user-avatar">👤</div>}
          </div>
        ))}

        {typing && (
          <div className="chat-bubble-wrapper assistant">
            <div className="chat-avatar bot-avatar">🤖</div>
            <div className="chat-bubble assistant">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-quick-replies">
        {quickReplies.map((q, i) => (
          <button key={i} className="quick-reply" onClick={() => handleSend(q)}>
            {q}
          </button>
        ))}
      </div>

      <div className="chat-input-bar">
        <input
          type="text"
          className="chat-input"
          placeholder="Ask about your health..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          id="chat-input"
        />
        <button
          className="chat-send-btn"
          onClick={() => handleSend()}
          disabled={!input.trim() || typing}
          id="chat-send"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
