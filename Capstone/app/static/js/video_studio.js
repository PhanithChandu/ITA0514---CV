// Video Detection Studio Module

class VideoStudio {
  constructor() {
    this.ws = null;
    this.videoEl = null;
    this.canvas = null;
    this.ctx = null;
    this.offscreen = null;
    this.offCtx = null;
    this.playing = false;
    this.rafId = null;
    this.lastSendTime = 0;
    this.mediaUrl = null;

    this.initUI();
  }

  initUI() {
    const dropZone = document.getElementById('video-dropzone');
    const fileInput = document.getElementById('video-input');
    this.videoEl = document.getElementById('video-player');
    this.canvas = document.getElementById('video-canvas');

    if (!dropZone || !fileInput || !this.videoEl || !this.canvas) return;

    this.ctx = this.canvas.getContext('2d', { alpha: false });

    this.offscreen = document.createElement('canvas');
    this.offscreen.width = 640;
    this.offscreen.height = 360;
    this.offCtx = this.offscreen.getContext('2d', { alpha: false, desynchronized: true });

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        this.loadVideoFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        this.loadVideoFile(e.target.files[0]);
      }
    });

    const playBtn = document.getElementById('video-play-btn');
    if (playBtn) {
      playBtn.addEventListener('click', () => this.togglePlay());
    }

    const confSlider = document.getElementById('video-conf-slider');
    if (confSlider) {
      confSlider.addEventListener('input', (e) => {
        document.getElementById('video-conf-val').textContent = parseFloat(e.target.value).toFixed(2);
        this.sendControl();
      });
    }

    const iouSlider = document.getElementById('video-iou-slider');
    if (iouSlider) {
      iouSlider.addEventListener('input', (e) => {
        document.getElementById('video-iou-val').textContent = parseFloat(e.target.value).toFixed(2);
        this.sendControl();
      });
    }
  }

  loadVideoFile(file) {
    if (!file.type.startsWith('video/')) {
      showToast('Please select a valid video file (MP4, WEBP, MOV)', 'error');
      return;
    }

    if (this.mediaUrl) URL.revokeObjectURL(this.mediaUrl);
    this.mediaUrl = URL.createObjectURL(file);
    this.videoEl.src = this.mediaUrl;
    this.videoEl.load();

    document.getElementById('video-workspace').classList.remove('hidden');
    this.connectWS();

    this.videoEl.onloadedmetadata = () => {
      this.canvas.width = this.videoEl.videoWidth || 640;
      this.canvas.height = this.videoEl.videoHeight || 360;
    };
  }

  connectWS() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.sendControl();
    };

    this.ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      this.renderFrame(data.image);
      this.updateStats(data.fps, data.inference_ms, data.detections);
    };
  }

  togglePlay() {
    if (!this.videoEl.src) return;

    const playBtn = document.getElementById('video-play-btn');
    if (this.videoEl.paused) {
      this.videoEl.play();
      this.playing = true;
      if (playBtn) playBtn.innerHTML = '<i data-lucide="pause" class="w-5 h-5"></i> Pause';
      lucide.createIcons();
      this.loop();
    } else {
      this.videoEl.pause();
      this.playing = false;
      if (playBtn) playBtn.innerHTML = '<i data-lucide="play" class="w-5 h-5"></i> Play Video';
      lucide.createIcons();
    }
  }

  loop(now) {
    if (!this.playing || this.videoEl.paused) return;
    this.rafId = requestAnimationFrame((t) => this.loop(t));

    if (now - this.lastSendTime < 30) return; // limit to ~30 fps transmission
    this.lastSendTime = now;

    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.videoEl.readyState >= 2) {
      this.offCtx.drawImage(this.videoEl, 0, 0, 640, 360);
      const conf = parseFloat(document.getElementById('video-conf-slider').value);
      this.offscreen.toBlob((blob) => {
        if (blob && this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(blob);
        }
      }, 'image/jpeg', 0.75);
    }
  }

  renderFrame(b64) {
    const img = new Image();
    img.onload = () => {
      this.ctx.drawImage(img, 0, 0, this.canvas.width, this.canvas.height);
    };
    img.src = `data:image/jpeg;base64,${b64}`;
  }

  updateStats(fps, ms, detections) {
    document.getElementById('video-stat-fps').textContent = `${fps} FPS`;
    document.getElementById('video-stat-ms').textContent = `${ms} ms`;
    document.getElementById('video-stat-count').textContent = `${detections ? detections.length : 0} objects`;

    if (detections && detections.length > 0) {
      this.appendLog(detections);
    }
  }

  appendLog(detections) {
    const logContainer = document.getElementById('video-event-log');
    if (!logContainer) return;

    const timeStr = new Date().toLocaleTimeString();
    const summary = {};
    detections.forEach(d => summary[d.label] = (summary[d.label] || 0) + 1);

    const labels = Object.entries(summary).map(([l, c]) => `${l} (×${c})`).join(', ');
    const row = document.createElement('div');
    row.className = 'log-item flex justify-between items-center text-xs';
    row.innerHTML = `<span class="font-mono text-zinc-500">${timeStr}</span> <span class="font-bold text-white capitalize">${labels}</span>`;

    logContainer.prepend(row);
    while (logContainer.children.length > 25) {
      logContainer.removeChild(logContainer.lastChild);
    }
  }

  sendControl() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const conf = parseFloat(document.getElementById('video-conf-slider').value);
    const iou = parseFloat(document.getElementById('video-iou-slider').value);
    this.ws.send(JSON.stringify({ conf, iou, enabled: true }));
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.videoStudio = new VideoStudio();
});
