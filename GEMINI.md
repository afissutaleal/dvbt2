# Project Rules & Architecture: DVB-T2 Broadcast AI Propagation & GIS Suite

Selamat datang di proyek **DVB-T2 Broadcast AI Propagation & GIS Suite**. File ini adalah panduan kontekstual otomatis untuk agen Google Antigravity dan tim pengembang.

---

## 📌 Ringkasan Proyek
Aplikasi berbasis Web Dashboard (**Streamlit**) yang mengintegrasikan Machine Learning (**XGBoost** + **Optuna**) dan Sistem Informasi Geografis (**Folium/Leaflet**) untuk memodelkan dan memprediksi kuat medan siaran televisi digital DVB-T2 (Band IV/V UHF) secara akurat hingga ke area *blank spot*.

Aplikasi ini mencakup:
1. **Tab 1:** Ingestion data fleksibel (Transmitter & multi-file Drive Test), deduplikasi spasial, kalkulasi EIRP empiris, dan alignment elevasi medan SRTM.
2. **Tab 2:** Multi-feature stratified 3-way split (70:15:15) dengan stratifikasi clearance (LoS vs NLoS), visualisasi 8 panel, tuning hyperparameter Bayesian Optuna, dan pelatihan XGBoost Regressor.
3. **Tab 3:** Peta cakupan poligon kotak grid ($100\text{ m}, 200\text{ m}, 500\text{ m}$) dengan 33 titik sub-sampling per kotak, kalkulasi median, dan saklar *Strict NLoS Cutoff*.
4. **Tab 4:** Kalkulator prediksi kuat medan titik tunggal (Single Point Predictor) dengan analisis elevasi medan, clearance, radio horizon, dan *Strict NLoS Cutoff*.

---

## 🏗️ Struktur & Modul Kode (`src/`)
* [`app.py`](./app.py): Entrypoint antarmuka dashboard interaktif Streamlit (Tab 1 s/d Tab 4).
* [`src/features.py`](./src/features.py): Definisi kolom fitur `MODEL_FEATURE_COLS` (8 fitur fisik) dan target `TARGET_COL`.
* [`src/preprocessor.py`](./src/preprocessor.py): Ingestion bebas nama kolom, deduplikasi spasial, perataan titik macet, perhitungan EIRP empiris, dan penggabungan clearance.
* [`src/terrain_engine.py`](./src/terrain_engine.py): Pengambilan elevasi medan (OpenTopography SRTM), kelengkungan bumi efektif ($k=4/3$), sampling adaptif $\le 500\text{ m}$ pada rentang $5\%-95\%$ lintasan, dan perhitungan clearance ($h_{\text{los}} - h_{\max}$).
* [`src/split_utils.py`](./src/split_utils.py): Partisi data latih/validasi/uji (70:15:15) berbasis stratifikasi multi-fitur (Jarak, RX AMSL, Azimuth, Clearance), tabel *sanity check*, dan gambar diagnostik 8 panel.
* [`src/model_engine.py`](./src/model_engine.py): Manajer model `DVBT2ModelManager` untuk optimasi Optuna, pelatihan XGBoost, penyimpanan model di `models/`, dan inferensi titik tunggal.
* [`src/grid_engine.py`](./src/grid_engine.py): Generator kotak poligon koordinat, sub-sampling 33 titik per kotak, batch inferensi vektor, kalkulasi median, dan filter *Strict NLoS Cutoff*.
* [`src/gis_viz.py`](./src/gis_viz.py): Peta interaktif Folium (OpenStreetMap, CartoDB, Satelit Esri), pewarnaan standar Band IV/V, serta grafik Parity Plot & Feature Importance.
* [`src/storage.py`](./src/storage.py): Modul penyimpanan persisten otomatis (`data/processed/{region_code}/`) untuk menyimpan data transmitter, dataset bersih, dan hasil split agar Tab 3 & Tab 4 langsung siap pakai saat aplikasi baru dibuka.

---

## 🔬 8 Fitur Fisik Model
1. `frequency_hz`: Frekuensi kerja siaran (Hz).
2. `eirp_dbm`: Daya pancar isotropis efektif empiris (dBm).
3. `distance_km`: Jarak geodetik Haversine pemancar-penerima (km).
4. `azimuth_sin`: $\sin(\theta_{\text{bearing}})$ sudut azimuth ($-1$ s.d. $1$).
5. `azimuth_cos`: $\cos(\theta_{\text{bearing}})$ sudut azimuth ($-1$ s.d. $1$).
6. `tx_antenna_amsl`: Tinggi antena pemancar di atas permukaan laut (m).
7. `rx_antenna_amsl`: Tinggi antena penerima di atas permukaan laut (m).
8. `clearance_m`: Jarak bebas lintasan terhadap puncak bukit penghalang ($h_{\text{los}} - h_{\max}$). Nilai $< 0$ menandakan kondisi terhalang kontur tanah (NLoS).

---

## 🛠️ Instruksi Pengembangan untuk Agen Antigravity
1. **Menjalankan Aplikasi:**
   ```powershell
   python -m streamlit run app.py
   # atau klik ganda run_app.bat
   ```
2. **Menjalankan Pengujian Lengkap:**
   Sebelum menyelesaikan tugas pengkodean, selalu jalankan skrip verifikasi otomatis:
   ```powershell
   python test_pipeline.py
   ```
   Pastikan seluruh 5 komponen pipeline (Preprocessing, Split, Training, Grid, Single Point) lolos dengan kode keluar 0.
3. **Kompatibilitas Fitur:**
   Ketika melakukan inferensi model XGBoost (`predict`), selalu periksa fitur yang didukung oleh model melalui `booster.feature_names` agar kompatibel ke belakang.
4. **Penyimpanan Persisten:**
   Gunakan fungsi-fungsi di [`src/storage.py`](./src/storage.py) untuk membaca dan memperbarui data wilayah yang tersimpan di `data/processed/{region_code}/`.
