// Live Webcam Stream Module

class WebcamStreamer {
  constructor() {
    this.ws = null;
    this.stream = null;
    this.videoEl = null;
    this.canvas = null;
    this.ctx = null;
    this.offscreen = null;
    this.offCtx = null;
    this.active = false;
    this.rafId = null;
    this.pendingSend = false;
    this.lastSendTime = 0;

    this.sendW = 640;
    this.sendH = 360;
    this.targetSendMs = 16;

    this.sendCount = 0;
    this.sendFpsTimer = 0;

    this.initUI();
  }

  initUI() {
    this.videoEl = document.getElementById('webcam-video');
    this.canvas = document.getElementById('webcam-canvas');

    if (!this.videoEl || !this.canvas) return;

    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.initOffscreen();

    const startBtn = document.getElementById('webcam-start-btn');
    const stopBtn = document.getElementById('webcam-stop-btn');
    const toggle = document.getElementById('webcam-toggle');

    if (startBtn) startBtn.addEventListener('click', () => this.start());
    if (stopBtn) stopBtn.addEventListener('click', () => this.stop());

    if (toggle) {
      toggle.addEventListener('change', () => this.sendControl());
    }

    const confSlider = document.getElementById('webcam-conf-slider');
    if (confSlider) {
      confSlider.addEventListener('input', (e) => {
        document.getElementById('webcam-conf-val').textContent = parseFloat(e.target.value).toFixed(2);
        this.sendControl();
      });
    }

    const iouSlider = document.getElementById('webcam-iou-slider');
    if (iouSlider) {
      iouSlider.addEventListener('input', (e) => {
        document.getElementById('webcam-iou-val').textContent = parseFloat(e.target.value).toFixed(2);
        this.sendControl();
      });
    }

    const resSelect = document.getElementById('webcam-res-select');
    if (resSelect) {
      resSelect.addEventListener('change', (e) => {
        const [w, h] = e.target.value.split('x').map(Number);
        this.sendW = w;
        this.sendH = h;
        this.offscreen.width = w;
        this.offscreen.height = h;
        this.targetSendMs = w <= 320 ? 12 : w <= 640 ? 16 : 20;
      });
    }
  }

  initOffscreen() {
    this.offscreen = document.createElement('canvas');
    this.offscreen.width = this.sendW;
    this.offscreen.height = this.sendH;
    this.offCtx = this.offscreen.getContext('2d', { alpha: false, desynchronized: true });
  }

  async start() {
    if (this.active) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 60, max: 120 },
          facingMode: 'user',
        },
        audio: false,
      });

      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();

      this.canvas.width = this.videoEl.videoWidth || 1280;
      this.canvas.height = this.videoEl.videoHeight || 720;

      this.active = true;
      this.connectWS();
      this.scheduleLoop();

      document.getElementById('webcam-start-btn').disabled = true;
      document.getElementById('webcam-stop-btn').disabled = false;
      document.getElementById('webcam-overlay').classList.add('hidden');

      showToast('Webcam stream started successfully!', 'success');
    } catch (err) {
      showToast('Camera access denied or error: ' + err.message, 'error');
    }
  }

  stop() {
    this.active = false;
    this.pendingSend = false;

    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.videoEl.srcObject = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    document.getElementById('webcam-start-btn').disabled = false;
    document.getElementById('webcam-stop-btn').disabled = true;
    document.getElementById('webcam-overlay').classList.remove('hidden');
    this.updateStatus(false);
  }

  connectWS() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.ws = new WebSocket(`${proto}://${location.host}/ws/detect`);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.updateStatus(true);
      this.sendControl();
    };

    this.ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      this.renderFrame(data.image);
      this.updateStats(data.fps, data.inference_ms, data.detections);
      this.updateLog(data.detections);
    };

    this.ws.onclose = () => {
      this.updateStatus(false);
      if (this.active) {
        setTimeout(() => this.connectWS(), 1500);
      }
    };
    this.ws.onerror = () => this.ws.close();
  }

  scheduleLoop() {
    if (!this.active) return;
    this.rafId = requestAnimationFrame((now) => this.captureLoop(now));
  }

  captureLoop(now) {
    if (!this.active) return;
    this.rafId = requestAnimationFrame((n) => this.captureLoop(n));

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (now - this.lastSendTime < this.targetSendMs) return;
    if (this.pendingSend) return;
    if (this.ws.bufferedAmount > 256 * 1024) return;
    if (this.videoEl.readyState < 2) return;

    this.lastSendTime = now;
    this.pendingSend = true;

    this.offCtx.drawImage(this.videoEl, 0, 0, this.sendW, this.sendH);
    const qual = parseFloat(document.getElementById('webcam-qual-slider').value) / 100;

    this.offscreen.toBlob((blob) => {
      this.pendingSend = false;
      if (!blob || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      this.ws.send(blob);

      this.sendCount++;
      if (now - this.sendFpsTimer >= 1000) {
        document.getElementById('webcam-stat-tx').textContent = `${this.sendCount} tx/s`;
        this.sendCount = 0;
        this.sendFpsTimer = now;
      }
    }, 'image/jpeg', qual);
  }

  renderFrame(b64) {
    createImageBitmap(new Blob([Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))], { type: 'image/jpeg' })).then(
      (bmp) => {
        if (this.canvas.width !== bmp.width) this.canvas.width = bmp.width;
        if (this.canvas.height !== bmp.height) this.canvas.height = bmp.height;
        this.ctx.drawImage(bmp, 0, 0);
        bmp.close();
      }
    );
  }

  updateStats(fps, ms, detections) {
    document.getElementById('webcam-stat-fps').textContent = `${fps} FPS`;
    document.getElementById('webcam-stat-ms').textContent = `${ms} ms`;
    document.getElementById('webcam-stat-count').textContent = `${detections ? detections.length : 0} objects`;
  }

  updateLog(detections) {
    if (!detections || detections.length === 0) return;
    const container = document.getElementById('webcam-event-log');
    if (!container) return;

    const timeStr = new Date().toLocaleTimeString();
    const agg = {};
    detections.forEach((d) => (agg[d.label] = (agg[d.label] || 0) + 1));

    const html = Object.entries(agg)
      .map(
        ([lbl, cnt]) =>
          `<div class="flex justify-between items-center text-xs py-1"><span class="font-bold text-white capitalize">${lbl}</span> <span class="px-2 py-0.5 rounded bg-orange-500/20 text-orange-400 font-mono font-bold">×${cnt}</span></div>`
      )
      .join('');

    const entry = document.createElement('div');
    entry.className = 'log-item';
    entry.innerHTML = `<div class="text-[0.65rem] font-mono text-zinc-500 font-bold mb-1">${timeStr}</div>${html}`;

    container.prepend(entry);
    while (container.children.length > 20) container.removeChild(container.lastChild);
  }

  sendControl() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const conf = parseFloat(document.getElementById('webcam-conf-slider').value);
    const iou = parseFloat(document.getElementById('webcam-iou-slider').value);
    const enabled = document.getElementById('webcam-toggle').checked;
    this.ws.send(JSON.stringify({ conf, iou, enabled }));
  }

  updateStatus(online) {
    const dot = document.getElementById('webcam-status-dot');
    const label = document.getElementById('webcam-status-label');
    if (dot) dot.className = 'status-dot' + (online ? '' : ' off');
    if (label) label.textContent = online ? 'Streaming' : 'Offline';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.webcamStreamer = new WebcamStreamer();
});
