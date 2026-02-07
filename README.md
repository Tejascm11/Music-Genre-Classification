# Music-Genre-Classification
Music Genre Classifier is a web-based application that classifies the genre of an
audio file using simple audio signal processing techniques. Users can upload
an audio file (MP3 or WAV), visualize its waveform, and receive a predicted
music genre based on extracted audio features.
---

## ⚙️ How It Works
1. User uploads an audio file (MP3/WAV)
2. Audio is decoded using the Web Audio API
3. Audio features are extracted:
   - Spectral Centroid
   - RMS Energy
   - Zero-Crossing Rate
4. A heuristic classifier predicts the music genre
5. Confidence and feature meters are displayed
6. Optionally, a TensorFlow.js model can be loaded for prediction

---


## 📄 License
This project is licensed under the MIT License.

---

## 👤 Author
Developed as an academic web-based audio classification project.

