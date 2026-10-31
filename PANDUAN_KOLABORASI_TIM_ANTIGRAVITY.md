# PANDUAN SERAH TERIMA & PENGEMBANGAN BERSAMA TIM
## DVB-T2 Broadcast AI Propagation & GIS Suite
### Menggunakan Google Antigravity (Metode Distribusi Berkas Arsip / ZIP)

---

Dokumen ini disusun sebagai panduan resmi serah terima teknis agar rekan tim dapat segera melanjutkan pengembangan aplikasi **DVB-T2 Broadcast AI Propagation & GIS Suite** di komputer masing-masing menggunakan platform **Google Antigravity**.

---

## DAFTAR ISI
1. [Langkah Pengirim (Pemilik Proyek Awal)](#1-langkah-pengirim-pemilik-proyek-awal)
2. [Langkah Penerima (Rekan Tim Pengembang)](#2-langkah-penerima-rekan-tim-pengembang)
3. [Cara Berkolaborasi dengan AI Antigravity](#3-cara-berkolaborasi-dengan-ai-antigravity)
4. [Peta Berkas & Modul Aplikasi](#4-peta-berkas--modul-aplikasi)
5. [Tanya Jawab & Penanganan Masalah (Troubleshooting)](#5-tanya-jawab--penanganan-masalah-troubleshooting)

---

## 1. LANGKAH PENGIRIM (Pemilik Proyek Awal)

Sebagai pihak yang membagikan proyek, lakukan langkah berikut untuk mengemas seluruh kode sumber, model AI, dan dataset:

### A. Bersihkan Berkas Sementara (Opsional)
Pastikan tidak ada berkas temporer yang tidak diperlukan. Buka PowerShell dan masuk ke direktori proyek:
```powershell
cd C:\Users\user\.gemini\antigravity\scratch\dvbt2-propagation-ai
Get-ChildItem -Filter "scratch_*" | Remove-Item -Force
```

### B. Buat Berkas Arsip ZIP
Jalankan perintah berikut di PowerShell untuk mengompresi proyek menjadi satu berkas ZIP di folder Downloads:
```powershell
Compress-Archive -Path "C:\Users\user\.gemini\antigravity\scratch\dvbt2-propagation-ai\*" -DestinationPath "C:\Users\user\Downloads\dvbt2-propagation-ai.zip" -Force
```
*(File arsip siap kirim saat ini sudah dibuatkan otomatis dan berada di: `C:\Users\user\Downloads\dvbt2-propagation-ai.zip`).*

### C. Kirimkan Berkas ZIP ke Tim
Unggah berkas `dvbt2-propagation-ai.zip` ke media berbagi tim:
* Google Drive / Microsoft OneDrive / Dropbox
* Flashdisk / Media Penyimpanan Eksternal
* Local Network Share (LAN)

---

## 2. LANGKAH PENERIMA (Rekan Tim Pengembang)

Sebagai rekan tim yang menerima berkas proyek, ikuti 5 langkah mudah berikut:

### Langkah 1: Ekstraksi Berkas ZIP
1. Unduh berkas `dvbt2-propagation-ai.zip` ke komputer Anda.
2. Klik kanan pada berkas zip tersebut > **Extract All...** (Ekstrak Semua).
3. Tentukan folder tujuan ekstraksi, misalnya:
   * `D:\Projects\dvbt2-propagation-ai` atau
   * `C:\Users\<Nama_Anda>\Projects\dvbt2-propagation-ai`

---

### Langkah 2: Buka Folder Proyek di Google Antigravity
1. Buka aplikasi **Google Antigravity** (baik Antigravity IDE maupun Antigravity Desktop).
2. Klik menu **File** > **Open Folder...** (atau kombinasi tombol `Ctrl + K, Ctrl + O`).
3. Pilih folder hasil ekstraksi: `dvbt2-propagation-ai`.
4. Folder proyek kini aktif sebagai **Active Workspace** di Antigravity.

---

### Langkah 3: Penyiapan Lingkungan Python & Instalasi Dependensi
Pastikan komputer Anda telah terpasang **Python 3.10** atau **Python 3.11** (rekomendasi Python 3.11 64-bit).

Buka **Terminal** di dalam Antigravity (menu **Terminal > New Terminal** atau tombol ``Ctrl + ` ``), lalu jalankan:

1. *(Sangat Disarankan)* Buat Virtual Environment agar paket tidak bentrok dengan proyek lain:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
   *(Tanda `(venv)` akan muncul di awal baris terminal).*

2. Instal seluruh paket dependensi melalui file `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```
   *Paket yang diinstal meliputi: `streamlit`, `xgboost`, `optuna`, `folium`, `streamlit-folium`, `geopy`, `scikit-learn`, `matplotlib`, `openpyxl`, `requests`, `pandas`, dan `numpy`.*

---

### Langkah 4: Uji Verifikasi Otomatis Kesiapan Sistem
Sebelum menjalankan aplikasi, jalankan skrip verifikasi otomatis yang telah disediakan:
```powershell
python test_pipeline.py
```
Skrip ini akan menguji 5 modul inti secara otomatis:
1. Ingestion Data, Deduplikasi & Kalkulasi EIRP.
2. Stratified 3-Way Split (70:15:15) dengan fitur Clearance.
3. Optimasi Hyperparameter (Optuna) & Pelatihan XGBoost.
4. Pembangkitan Kotak Grid ($100\text{ m} \times 100\text{ m}$) & Sub-sampling 33 Titik.
5. Prediksi Titik Tunggal (Single Point Predictor) & Filter Radio Horizon.

Jika terminal menampilkan:
```text
🎉 ALL 5 PIPELINE COMPONENTS TESTED & VALIDATED SUCCESSFULLY!
```
Maka lingkungan kerja Anda **100% siap** dan tidak ada kendala dependensi maupun algoritma.

---

### Langkah 5: Menjalankan Aplikasi Web Dashboard
Untuk membuka aplikasi antarmuka pengguna:
* **Cara Cepat:** Klik ganda file `run_app.bat` di dalam folder proyek.
* **Atau melalui Terminal:**
  ```powershell
  python -m streamlit run app.py
  ```
Browser web akan otomatis terbuka di alamat:
```text
http://localhost:8501
```

> **Catatan Penting:**
> Berkat fitur **Penyimpanan Persisten (*Persistent Storage*)** yang telah disertakan di folder `data/processed/`, ketika aplikasi pertama kali dibuka, data pemancar dan model AI wilayah **Balikpapan - Kaltim 1** langsung aktif secara otomatis! Anda bisa langsung mencoba **Tab 3 (Peta Coverage Grid)** dan **Tab 4 (Prediksi Titik Tunggal)** tanpa perlu mengunggah ulang data.

---

## 3. CARA BERKOLABORASI DENGAN AI ANTIGRAVITY

Salah satu keunggulan terbesar menggunakan Google Antigravity untuk proyek ini adalah tersedianya file **[`GEMINI.md`](./GEMINI.md)** di *root directory*.

### Mengapa Ini Sangat Membantu?
File `GEMINI.md` berfungsi sebagai instruksi kontekstual otomatis. Ketika Anda membuka chat dengan Antigravity di workspace ini, **agen Antigravity secara otomatis telah membaca dan memahami:**
* Konsep 8 fitur fisik model propagasi (termasuk *clearance* medan SRTM).
* Aturan kriteria kuat medan TV digital DVB-T2 Band IV/V UHF (ITU-R BT.1368).
* Logika inferensi median sub-titik dan saklar *Strict NLoS Cutoff*.
* Struktur file dan fungsi di dalam direktori `src/`.

### Contoh Perintah / Prompt yang Bisa Anda Berikan ke Antigravity:
Anda tidak perlu lagi mengajari Antigravity dari awal. Anda bisa langsung meminta pengembangan fitur baru:

* **Contoh 1 (Ekspor Format SIG):**
  > *"Antigravity, tolong tambahkan tombol di Tab 3 untuk mengunduh hasil poligon kotak coverage ke format file Shapefile (.shp) atau GeoJSON agar bisa dibuka di QGIS/ArcGIS."*

* **Contoh 2 (Peningkatan Visualisasi Tab 4):**
  > *"Antigravity, di Tab 4 tolong tambahkan grafik visual irisan kontur tanah 2D (Terrain Elevation Cross-Section Profile) dari koordinat pemancar ke titik penerima."*

* **Contoh 3 (Dukungan Wilayah Layanan Baru):**
  > *"Antigravity, saya punya data transmitter dan drive test untuk wilayah Surabaya / Jatim 1. Tolong bantu verifikasi format datanya dan jalankan proses ingestion di Tab 1."*

* **Contoh 4 (Refactoring / Optimasi):**
  > *"Antigravity, tolong periksa fungsi batch prediction di grid_engine.py, apakah ada bagian yang bisa dioptimasi menggunakan multiprocessing atau vektorisasi NumPy agar lebih cepat pada radius 50 km."*

---

## 4. PETA BERKAS & MODUL APLIKASI

Berikut adalah fungsi masing-masing komponen di dalam repositori:

| Berkas / Direktori | Deskripsi & Tanggung Jawab Modul |
|---|---|
| `app.py` | Antarmuka utama Streamlit Dashboard (Tab 1 s/d Tab 4). |
| `GEMINI.md` | Konfigurasi aturan proyek otomatis untuk agen Google Antigravity. |
| `run_app.bat` | Skrip pintasan 1-klik untuk menjalankan Streamlit di Windows. |
| `test_pipeline.py` | Skrip uji otomatis menyeluruh (*end-to-end regression test*). |
| `requirements.txt` | Daftar pustaka Python resmi beserta versinya. |
| `src/features.py` | Definisi 8 fitur fisik (`MODEL_FEATURE_COLS`) dan nama target. |
| `src/preprocessor.py` | Pembersihan data, deduplikasi spasial, agregasi titik macet, dan estimasi EIRP. |
| `src/terrain_engine.py` | Pengambilan elevasi medan SRTM, kurvatur bumi $k=4/3$, sampling adaptif $\le 500\text{ m}$, dan hitung clearance lintasan. |
| `src/split_utils.py` | Stratified 3-way split (70:15:15) berbasis clearance, gambar komparatif 8 panel, dan tabel sanity check. |
| `src/model_engine.py` | XGBoost Regressor engine, optimasi Optuna Bayesian TPE, kalkulator titik tunggal, dan horizon radio. |
| `src/grid_engine.py` | Generator grid poligon ($100\text{ m}, 200\text{ m}, 500\text{ m}$), sub-sampling 33 titik, batch inference, dan Strict NLoS cutoff. |
| `src/gis_viz.py` | Renderer peta interaktif Folium (Leaflet), palet warna ITU Band IV/V, Parity Plot, dan grafik Feature Importance. |
| `src/storage.py` | Mesin penyimpanan persisten lokal di `data/processed/{region_code}/` (*zero re-upload*). |
| `data/` | Dataset contoh transmitter & pengukuran drive test lapangan (Balikpapan). |
| `models/` | Berkas model terlatih JSON (`*_xgb_model.json`) dan metrik evaluasi. |

---

## 5. TANYA JAWAB & PENANGANAN MASALAH (TROUBLESHOOTING)

### Q1: Muncul error `pip install` terkait C++ compiler atau build wheel?
**Solusi:** Pastikan Anda menggunakan Python versi 3.10 atau 3.11 64-bit resmi dari [python.org](https://www.python.org/downloads/). Seluruh pustaka (`xgboost`, `scikit-learn`, `numpy`, `pandas`) telah menyediakan binary pre-compiled (*wheel*) untuk Windows 64-bit sehingga tidak memerlukan instalasi Visual C++ Compiler manual.

### Q2: Port 8501 sudah digunakan oleh proses lain?
**Solusi:** Jalankan Streamlit dengan menentukan port alternatif:
```powershell
python -m streamlit run app.py --server.port 8502
```

### Q3: Bagaimana jika pengambilan elevasi medan (SRTM) lambat atau tidak ada koneksi internet?
**Solusi:** Modul `terrain_engine.py` memiliki mekanisme fallback otomatis (interpolasi ketinggian TX-RX atau model medan sintetis realistis) jika API OpenTopography/SRTM sedang offline, sehingga aplikasi tetap dapat berjalan normal untuk pengujian.

### Q4: Apakah anggota tim bisa menambahkan algoritma AI lain selain XGBoost?
**Solusi:** Sangat bisa! Rekan tim dapat meminta Antigravity membuatkan pembanding model (seperti LightGBM, CatBoost, atau Random Forest) di dalam modul `src/model_engine.py`.

---

*Dokumen disusun pada: Oktober 2026*  
*Tim Pengembang: DVB-T2 Broadcast AI Propagation & GIS Suite*
