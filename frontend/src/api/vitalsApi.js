/**
 * API service layer — all fetch calls to the Flask backend.
 */

const BASE = '/api';

async function request(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  
  let data;
  try {
    data = await res.json();
  } catch (err) {
    if (!res.ok) {
      throw new Error(`Server error (${res.status}). Is the backend running?`);
    }
    throw new Error('Failed to parse server response');
  }

  if (!res.ok && !data.errors) throw new Error(data.error || 'Request failed');
  return data;
}

/** POST /api/vitals — submit vitals, run full agent pipeline */
export async function submitVitals(formData) {
  return request('/vitals', {
    method: 'POST',
    body: JSON.stringify(formData),
  });
}

/** GET /api/history/:userId — 7-day trend data */
export async function getHistory(userId) {
  return request(`/history/${userId}`);
}

/** GET /api/report/:recordId — structured report data */
export async function getReport(recordId) {
  return request(`/report/${recordId}`);
}

/** POST /api/chat — chat assistant message */
export async function sendChat(message, userId) {
  return request('/chat', {
    method: 'POST',
    body: JSON.stringify({ message, user_id: userId }),
  });
}

/** GET /api/doctors/:recordId — doctor suggestions */
export async function getDoctors(recordId) {
  return request(`/doctors/${recordId}`);
}

/** GET /api/recommendations/:recordId — health recommendations */
export async function getRecommendations(recordId) {
  return request(`/recommendations/${recordId}`);
}

/** GET /api/ui/config — full UI configuration from Agent 5 */
export async function getUIConfig() {
  return request('/ui/config');
}

/** GET /api/status — system health from Agent 6 */
export async function getStatus() {
  return request('/status');
}
