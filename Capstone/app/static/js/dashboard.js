// Global Navigation & Dashboard Manager

function switchTab(tabId) {
  // Update nav buttons
  document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));
  const targetNav = document.getElementById('nav-' + tabId);
  if (targetNav) targetNav.classList.add('active');

  // Hide all sections
  document.querySelectorAll('.page-content').forEach(page => page.classList.remove('active'));
  
  // Show target section
  const targetPage = document.getElementById('page-' + tabId);
  if (targetPage) targetPage.classList.add('active');

  // Trigger stop on webcam stream if moving away from webcam tab
  if (tabId !== 'webcam' && window.webcamStreamer && window.webcamStreamer.active) {
    window.webcamStreamer.stop();
  }
}

// Global Toast Notifications
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  const bg = type === 'error' ? 'bg-red-900/90 border-red-700' : type === 'success' ? 'bg-emerald-900/90 border-emerald-700' : 'bg-orange-900/90 border-orange-700';
  toast.className = `px-4 py-3 rounded-xl border text-white text-sm shadow-xl flex items-center gap-3 transform transition-all duration-300 translate-y-2 opacity-0 ${bg}`;
  toast.innerHTML = `<span>${message}</span>`;
  
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  }, 10);

  setTimeout(() => {
    toast.classList.add('opacity-0', 'translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// Health Check Monitor
async function checkBackendHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const statusBadge = document.getElementById('system-status');
    if (statusBadge) {
      statusBadge.innerHTML = `<span class="status-dot"></span> Server Ready (${data.model})`;
    }
  } catch (err) {
    const statusBadge = document.getElementById('system-status');
    if (statusBadge) {
      statusBadge.innerHTML = `<span class="status-dot off"></span> Server Offline`;
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  setInterval(checkBackendHealth, 10000);
});
