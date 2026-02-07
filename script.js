
let audioCtx, bufferSource;
let audioBufferGlobal = null;

// DOM elements
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const analyzeBtn = document.getElementById('analyzeBtn');
const playBtn = document.getElementById('playBtn');
const stopBtn = document.getElementById('stopBtn');
const fileName = document.getElementById('fileName');
const wave = document.getElementById('wave');
const ctx = wave.getContext('2d');
const debug = document.getElementById('debug');
const predLabel = document.getElementById('predLabel');
const confText = document.getElementById('confText');
const centroidMeter = document.getElementById('centroidMeter');
const rmsMeter = document.getElementById('rmsMeter');
const zcrMeter = document.getElementById('zcrMeter');
const modelUrlInput = document.getElementById('modelUrl');

// ===== Drag & Drop setup =====
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
  });
});

dropZone.addEventListener('drop', async e => {
  const f = e.dataTransfer.files[0];
  if (f) await loadFile(f);
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', async e => {
  if (e.target.files[0]) await loadFile(e.target.files[0]);
});

// ===== Load Audio File =====
async function loadFile(file) {
  fileName.textContent = `${file.name} (${Math.round(file.size / 1024)}KB)`;
  const array = await file.arrayBuffer();
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  audioBufferGlobal = await audioCtx.decodeAudioData(array.slice(0));
  analyzeBtn.disabled = false;
  playBtn.disabled = false;
  stopBtn.disabled = false;
  drawWaveform(audioBufferGlobal);
}

// ===== Draw Waveform =====
function drawWaveform(buffer) {
  const chan = buffer.getChannelData(0);
  const step = Math.ceil(chan.length / wave.width);
  ctx.clearRect(0, 0, wave.width, wave.height);
  ctx.fillStyle = '#021428';
  ctx.fillRect(0, 0, wave.width, wave.height);
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = 'rgba(255,255,255,0.7)';
  ctx.beginPath();
  const mid = wave.height / 2;
  for (let i = 0; i < wave.width; i++) {
    const start = i * step;
    let sum = 0;
    for (let j = 0; j < step && (start + j) < chan.length; j++) sum += Math.abs(chan[start + j]);
    const v = sum / step;
    const y = mid - v * mid * 1.8;
    if (i === 0) ctx.moveTo(i, y); else ctx.lineTo(i, y);
  }
  ctx.stroke();
}

// ===== Audio Playback =====
function playBuffer() {
  if (!audioBufferGlobal) return;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  bufferSource = audioCtx.createBufferSource();
  bufferSource.buffer = audioBufferGlobal;
  bufferSource.connect(audioCtx.destination);
  bufferSource.start();
}

function stopBuffer() {
  if (bufferSource) {
    try { bufferSource.stop(); } catch (e) { }
    bufferSource = null;
  }
}

playBtn.addEventListener('click', () => playBuffer());
stopBtn.addEventListener('click', () => stopBuffer());

// ===== Analyze Button =====
analyzeBtn.addEventListener('click', async () => {
  if (!audioBufferGlobal) return;
  debug.textContent = 'Computing features...';
  const features = await computeFeatures(audioBufferGlobal);
  debug.textContent = JSON.stringify(features, null, 2);

  const modelUrl = modelUrlInput.value.trim();
  if (modelUrl) {
    confText.textContent = 'Using TF.js model...';
    try {
      await loadAndPredictWithTf(modelUrl, features);
      return;
    } catch (err) {
      console.warn('TF.js model failed:', err);
      confText.textContent = 'TF.js model failed — using heuristic';
    }
  }

  const out = heuristicClassifier(features);
  showPrediction(out, features);
});

// ===== Compute Audio Features =====
async function computeFeatures(buffer) {
  const sampleRate = buffer.sampleRate;
  const channelData = buffer.getChannelData(0);
  const frameSize = 2048, hop = 1024;

  const fft = arr => {
    const N = arr.length;
    const mags = new Float32Array(N / 2);
    for (let k = 0; k < N / 2; k++) {
      let re = 0, im = 0;
      for (let n = 0; n < N; n++) {
        const phi = (2 * Math.PI * k * n) / N;
        re += arr[n] * Math.cos(phi);
        im -= arr[n] * Math.sin(phi);
      }
      mags[k] = Math.sqrt(re * re + im * im);
    }
    return mags;
  };

  let centroidSum = 0, rmsSum = 0, zcrSum = 0, frames = 0;
  for (let i = 0; i + frameSize < channelData.length; i += hop) {
    const frame = channelData.slice(i, i + frameSize);
    for (let n = 0; n < frame.length; n++) {
      frame[n] *= 0.5 * (1 - Math.cos((2 * Math.PI * n) / (frame.length - 1)));
    }

    // RMS
    let s = 0;
    for (let n = 0; n < frame.length; n++) s += frame[n] * frame[n];
    const rms = Math.sqrt(s / frame.length);
    rmsSum += rms;

    // ZCR
    let z = 0;
    for (let n = 1; n < frame.length; n++) if (frame[n] * frame[n - 1] < 0) z++;
    const zcr = z / frame.length;
    zcrSum += zcr;

    // Spectral Centroid
    const mags = fft(frame);
    let num = 0, den = 1e-9;
    for (let k = 0; k < mags.length; k++) {
      num += k * mags[k];
      den += mags[k];
    }
    const centroidBin = num / den;
    const centroidHz = centroidBin * (sampleRate / frameSize);
    centroidSum += centroidHz;
    frames++;
    if (frames > 60) break;
  }

  const centroidMean = centroidSum / frames;
  const rmsMean = rmsSum / frames;
  const zcrMean = zcrSum / frames;
  const centroidNorm = Math.min(1, centroidMean / (sampleRate / 2));

  return { centroidHz: Math.round(centroidMean), centroid: centroidNorm, rms: rmsMean, zcr: zcrMean };
}

// ===== Heuristic Classifier =====
function heuristicClassifier(f) {
  const scores = { rock: 0, classical: 0, hiphop: 0, jazz: 0 };

  scores.rock += f.centroid * 1.0 + f.rms * 1.5 + f.zcr * 0.8;
  scores.classical += (1 - f.centroid) * 1.2 + (1 - f.zcr) * 1.0 + f.rms * 0.6;
  scores.hiphop += f.centroid * 0.6 + f.rms * 2.0 + f.zcr * 1.2;
  scores.jazz += (1 - f.centroid) * 0.3 + f.rms * 0.9 + f.zcr * 0.6;

  const vals = Object.values(scores);
  const min = Math.min(...vals);
  if (min < 0) for (let k in scores) scores[k] -= min;

  const sum = Object.values(scores).reduce((a, b) => a + b, 0) + 1e-9;
  const probs = {};
  for (const k in scores) probs[k] = scores[k] / sum;

  const best = Object.keys(probs).sort((a, b) => probs[b] - probs[a])[0];
  return { best, probs };
}

// ===== Show Prediction =====
function showPrediction(out, f) {
  predLabel.textContent = out.best.toUpperCase();
  confText.textContent = 'Confidence: ' + Math.round(out.probs[out.best] * 100) + '%';
  centroidMeter.style.width = Math.round(f.centroid * 100) + '%';
  rmsMeter.style.width = Math.min(100, Math.round(f.rms * 100)) + '%';
  zcrMeter.style.width = Math.round(Math.min(1, f.zcr) * 100) + '%';
}

// ===== Optional TF.js Model Support =====
async function loadAndPredictWithTf(modelUrl, features) {
  if (!window.tf) {
    const s = document.createElement('script');
    s.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.12.0/dist/tf.min.js';
    document.head.appendChild(s);
    await new Promise((res, rej) => { s.onload = res; s.onerror = rej; });
  }

  const model = await tf.loadLayersModel(modelUrl);
  const input = tf.tensor2d([[features.centroid, features.rms, features.zcr]]);
  const logits = model.predict(input);
  const probs = await logits.data();
  const labels = ['rock', 'classical', 'hiphop', 'jazz'];

  let bestIdx = 0, best = probs[0], sum = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > best) { best = probs[i]; bestIdx = i; }
    sum += probs[i];
  }

  const normalized = {};
  for (let i = 0; i < labels.length; i++) normalized[labels[i]] = probs[i] / sum;
  showPrediction({ best: labels[bestIdx], probs: normalized }, features);
}