// Static Image Studio Module

class ImageStudio {
  constructor() {
    this.currentFile = null;
    this.lastResult = null;
    this.initUI();
  }

  initUI() {
    const dropZone = document.getElementById('image-dropzone');
    const fileInput = document.getElementById('image-input');
    const confSlider = document.getElementById('image-conf-slider');
    const iouSlider = document.getElementById('image-iou-slider');
    const downloadBtn = document.getElementById('btn-download-image');
    const exportBtn = document.getElementById('btn-export-json');

    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        this.loadImageFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        this.loadImageFile(e.target.files[0]);
      }
    });

    if (confSlider) {
      confSlider.addEventListener('input', (e) => {
        document.getElementById('image-conf-val').textContent = parseFloat(e.target.value).toFixed(2);
      });
      confSlider.addEventListener('change', () => this.processImage());
    }

    if (iouSlider) {
      iouSlider.addEventListener('input', (e) => {
        document.getElementById('image-iou-val').textContent = parseFloat(e.target.value).toFixed(2);
      });
      iouSlider.addEventListener('change', () => this.processImage());
    }

    if (downloadBtn) {
      downloadBtn.addEventListener('click', () => this.downloadAnnotatedImage());
    }

    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportJSON());
    }
  }

  loadImageFile(file) {
    if (!file.type.startsWith('image/')) {
      showToast('Please select a valid image file (JPG, PNG, WEBP)', 'error');
      return;
    }
    this.currentFile = file;

    // Display original preview
    const reader = new FileReader();
    reader.onload = (e) => {
      const origImg = document.getElementById('image-orig-preview');
      if (origImg) origImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    document.getElementById('image-workspace').classList.remove('hidden');
    this.processImage();
  }

  async processImage() {
    if (!this.currentFile) return;

    const conf = document.getElementById('image-conf-slider').value;
    const iou = document.getElementById('image-iou-slider').value;

    const formData = new FormData();
    formData.append('file', this.currentFile);

    const spinner = document.getElementById('image-spinner');
    if (spinner) spinner.classList.remove('hidden');

    try {
      const res = await fetch(`/api/detect/image?conf=${conf}&iou=${iou}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Detection failed');

      const data = await res.json();
      this.lastResult = data;

      // Update annotated image
      const resultImg = document.getElementById('image-result-preview');
      if (resultImg) {
        resultImg.src = `data:image/jpeg;base64,${data.image}`;
      }

      // Metrics
      document.getElementById('img-stat-time').textContent = `${data.inference_ms} ms`;
      document.getElementById('img-stat-objects').textContent = `${data.total_objects} detected`;
      document.getElementById('img-stat-dims').textContent = `${data.width} × ${data.height}`;

      // Render summary tags
      this.renderSummary(data.summary);

      // Render detections list
      this.renderDetections(data.detections);

      showToast(`Detected ${data.total_objects} objects in ${data.inference_ms}ms`, 'success');
    } catch (err) {
      showToast('Failed to process image: ' + err.message, 'error');
    } finally {
      if (spinner) spinner.classList.add('hidden');
    }
  }

  renderSummary(summary) {
    const container = document.getElementById('image-class-summary');
    if (!container) return;
    container.innerHTML = '';

    if (Object.keys(summary).length === 0) {
      container.innerHTML = '<span class="text-sm text-zinc-500">No objects detected.</span>';
      return;
    }

    Object.entries(summary).forEach(([label, count]) => {
      const tag = document.createElement('div');
      tag.className = 'px-3 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-xs font-semibold text-white flex items-center gap-2';
      tag.innerHTML = `<span class="w-2.5 h-2.5 rounded-full bg-orange-500"></span> <span class="capitalize">${label}</span> <span class="px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400 font-bold">${count}</span>`;
      container.appendChild(tag);
    });
  }

  renderDetections(detections) {
    const list = document.getElementById('image-det-list');
    if (!list) return;
    list.innerHTML = '';

    if (detections.length === 0) {
      list.innerHTML = '<p class="text-center text-sm text-zinc-500 py-8">No objects match the current confidence threshold.</p>';
      return;
    }

    detections.forEach((det, idx) => {
      const confPct = Math.round(det.confidence * 100);
      const row = document.createElement('div');
      row.className = 'p-3 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-between text-xs hover:border-orange-500/50 transition-colors';
      row.innerHTML = `
        <div class="flex items-center gap-3">
          <span class="w-6 h-6 rounded-lg bg-zinc-800 text-zinc-400 font-mono font-bold flex items-center justify-center">${idx + 1}</span>
          <div>
            <div class="font-bold text-white capitalize text-sm">${det.label}</div>
            <div class="text-zinc-500 font-mono text-[0.7rem]">BBox: [${det.bbox.join(', ')}]</div>
          </div>
        </div>
        <div class="text-right">
          <div class="font-bold text-orange-400 font-mono text-sm">${confPct}%</div>
          <div class="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-1">
            <div class="h-full bg-orange-500 rounded-full" style="width: ${confPct}%"></div>
          </div>
        </div>
      `;
      list.appendChild(row);
    });
  }

  downloadAnnotatedImage() {
    if (!this.lastResult || !this.lastResult.image) return;
    const a = document.createElement('a');
    a.href = `data:image/jpeg;base64,${this.lastResult.image}`;
    a.download = `yolo_detected_${Date.now()}.jpg`;
    a.click();
    showToast('Downloaded annotated image!', 'success');
  }

  exportJSON() {
    if (!this.lastResult) return;
    const jsonStr = JSON.stringify(this.lastResult, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `yolo_detections_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Exported detections JSON!', 'success');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.imageStudio = new ImageStudio();
});
