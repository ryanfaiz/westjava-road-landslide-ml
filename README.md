# Pemodelan Kerentanan dan Paparan Longsor Koridor Jalan Provinsi Jawa Barat

Sistem pemodelan spasial berbasis *Machine Learning* dan purwarupa **WebGIS interaktif** untuk memetakan tingkat kerentanan tanah longsor serta mengestimasi panjang segmen jalan terpapar (*road exposure assessment*) di sepanjang **297 ruas koridor jalan provinsi Jawa Barat, Indonesia**.

Proyek ini dikembangkan sebagai bagian dari mata kuliah **Pembelajaran Mesin (TEK1302)**, Program Studi **Teknologi Rekayasa Komputer, Sekolah Vokasi IPB University**.

---

## Daftar Isi
- [Pemodelan Kerentanan dan Paparan Longsor Koridor Jalan Provinsi Jawa Barat](#pemodelan-kerentanan-dan-paparan-longsor-koridor-jalan-provinsi-jawa-barat)
  - [Daftar Isi](#daftar-isi)
  - [1. Latar Belakang dan Urgensi](#1-latar-belakang-dan-urgensi)
  - [2. Tujuan Proyek](#2-tujuan-proyek)
  - [3. Sumber Dataset dan Variabel](#3-sumber-dataset-dan-variabel)
  - [4. Metodologi dan Pipeline Pemodelan](#4-metodologi-dan-pipeline-pemodelan)
    - [A. Preprocessing \& Feature Engineering](#a-preprocessing--feature-engineering)
    - [B. Temuan Exploratory Data Analysis (EDA)](#b-temuan-exploratory-data-analysis-eda)
    - [C. Skema Validasi Spasial](#c-skema-validasi-spasial)
  - [5. Hasil Evaluasi dan Perbandingan Model](#5-hasil-evaluasi-dan-perbandingan-model)
    - [Alasan Pemilihan Model Terbaik (`Logistic Regression`)](#alasan-pemilihan-model-terbaik-logistic-regression)
  - [6. Struktur Repositori dan Instalasi](#6-struktur-repositori-dan-instalasi)
    - [Struktur Direktori](#struktur-direktori)
    - [Panduan Instalasi \& Menjalankan Aplikasi](#panduan-instalasi--menjalankan-aplikasi)
  - [7. Keterbatasan dan Pengembangan Lanjutan](#7-keterbatasan-dan-pengembangan-lanjutan)
  - [8. Referensi Utama](#8-referensi-utama)
  - [9. Tim Pengembang](#9-tim-pengembang)

---

## 1. Latar Belakang dan Urgensi

* **Kondisi Eksisting:** Jaringan jalan provinsi Jawa Barat (297 segmen) merupakan urat nadi transportasi dan logistik regional yang banyak melintasi bentang alam pegunungan terjal dengan intensitas curah hujan tropis musiman yang tinggi.
* **Permasalahan Utama:** Aktivitas pemotongan kaki lereng (*road cut*) saat konstruksi jalan dan tingkat pelapukan batuan yang tinggi merusak kesetimbangan gaya penahan lereng, memicu bencana tanah longsor yang berulang kali memutus aksesibilitas wilayah.
* **Tantangan Metodologis:** Pemodelan spasial dengan pembagian data acak (*random split*) rentan mengalami *spatial data leakage* akibat autokorelasi spasial antar-titik yang berdekatan, sehingga menghasilkan metrik evaluasi yang bias dan terlalu optimistis.
* **Urgensi:** Pemangku kebijakan membutuhkan sistem pendukung keputusan berbasis data yang objektif untuk memetakan indeks kerentanan serta mengkuantifikasi total kilometer ruas jalan rawan sebagai dasar prioritas mitigasi infrastruktur.

---

## 2. Tujuan Proyek

1. **Komparasi Algoritma Klasifikasi:** Membangun dan membandingkan performa 4 algoritma *Machine Learning* (**Logistic Regression**, **Random Forest**, **XGBoost**, dan **Support Vector Machine**) pada zona penyangga (*buffer*) jalan sejauh $500\text{ m}$.
2. **Validasi Spasial Bebas Data Leakage:** Menerapkan skema **5-Fold Spatial Block Cross-Validation** berbasis klaster koordinat *KMeans* untuk menguji daya generalisasi model pada wilayah geografis yang benar-benar asing.
3. **Analisis Faktor Pemicu:** Mengukur kontribusi variabel topografi dan hidro-meteorologi serta memastikan independensi fitur melalui uji multikolinearitas (*Variance Inflation Factor* / VIF).
4. **Hilirisasi WebGIS Interaktif:** Mengembangkan purwarupa aplikasi berbasis **FastAPI** dan **HTML/JavaScript** untuk visualisasi peta zonasi kerentanan dan kalkulasi panjang ruas jalan terpapar ($\text{km}$).

---

## 3. Sumber Dataset dan Variabel

Ekstraksi spasial dilakukan pada radius *buffer* $500\text{ m}$ di sekitar 297 segmen jalan provinsi Jawa Barat, menghasilkan **651 titik sampel bersih** ($0$ *missing values*, $0$ duplikat koordinat).

| Layer / Variabel | Sumber Resmi | Peran | Karakteristik Statistik |
| :--- | :--- | :--- | :--- |
| **Jaringan Jalan Provinsi** | [SIGI Kementerian PUPR](https://sigi.pu.go.id/astv2/) | Batas Spasial | $297$ segmen jalan provinsi dengan *buffer* $500\text{ m}$ |
| **Elevation (Ketinggian)** | [Copernicus DEM 90m](https://dataspace.copernicus.eu/) | Fitur Prediktor | Mean: $463.27 \pm 359.70\text{ m}$ |
| **Slope (Kemiringan Lereng)** | [Copernicus DEM 90m](https://dataspace.copernicus.eu/) | Fitur Prediktor | Mean: $4.03^\circ \pm 3.87^\circ$ (Maks: $21.87^\circ$) |
| **Aspect (Arah Hadap Lereng)** | [Copernicus DEM 90m](https://dataspace.copernicus.eu/) | Fitur Sirkular | Mean: $176.42^\circ \pm 109.91^\circ$ ($0.08^\circ - 359.83^\circ$) |
| **Rainfall (Curah Hujan)** | [CHIRPS v2.0 (UCSB)](https://data.chc.ucsb.edu/products/CHIRPS/v2.0/) | Fitur Prediktor | Mean: $3009.88 \pm 398.36\text{ mm/thn}$ ($1981 - 2024$) |
| **Zonasi Gerak Tanah (ZKGT)** | [PVMBG Kementerian ESDM](https://vsi.esdm.go.id/portalmbg/) | **Target Biner** | Kelas `0` (Aman / Sangat Rendah) vs Kelas `1` (Rawan) |

---

## 4. Metodologi dan Pipeline Pemodelan

### A. Preprocessing & Feature Engineering
1. **Reproyeksi Spasial:** Seluruh layer vektor dan raster dikonversi ke sistem koordinat metrik **WGS 84 / UTM Zone 48S (`EPSG:32748`)** agar perhitungan jarak *buffer* $500\text{ m}$ dan panjang segmen ($\text{km}$) presisi.
2. **Transformasi Fitur Sirkular:** Variabel sudut arah hadap lereng (*aspect*) ditransformasikan menjadi dua komponen trigonometri untuk menjaga kontinuitas sudut antara $0^\circ$ dan $360^\circ$:
   $$\text{aspect\_sin} = \sin\left(\frac{\text{aspect} \times \pi}{180}\right), \quad \text{aspect\_cos} = \cos\left(\frac{\text{aspect} \times \pi}{180}\right)$$
3. **Uji Multikolinearitas (VIF):** Seluruh 5 fitur numerik memiliki nilai $\text{VIF} < 1.50$, membuktikan tidak adanya multikolinearitas antar-prediktor:
   * `elevation`: $1.1716$
   * `slope`: $1.4550$
   * `rainfall`: $1.2990$
   * `aspect_sin`: $1.0226$
   * `aspect_cos`: $1.0050$
4. **Standardisasi Fitur:** Penerapan `StandardScaler` dijalankan secara ketat di dalam setiap *fold* pelatihan untuk mencegah kebocoran distribusi data validasi.

### B. Temuan Exploratory Data Analysis (EDA)
* Variabel dengan korelasi linear positif tertinggi terhadap kelas target kerentanan adalah **Slope** ($r = 0.59$), diikuti oleh **Rainfall** ($r = 0.49$) dan **Elevation** ($r = 0.39$).
* Terdapat kontras distribusi yang jelas antara zona rawan dan stabil, dengan median kemiringan lereng pada kelas longsor sebesar $5.3^\circ$ dibandingkan $0.9^\circ$ pada kelas non-longsor.

### C. Skema Validasi Spasial
Dataset dipartisi menjadi **5 blok geografis** menggunakan algoritma *KMeans* berdasarkan koordinat spasial $(X, Y)$ UTM Zone 48S. Model dilatih pada 4 blok wilayah dan diuji pada 1 blok wilayah yang terpisah secara geografis.

---

## 5. Hasil Evaluasi dan Perbandingan Model

Evaluasi menggunakan **5-Fold Spatial Block Cross-Validation** menghasilkan perbandingan performa sebagai berikut:

| Algoritma Model | ROC-AUC (Mean; $\pm$ Std) | F1-Score (Mean) | Accuracy (Mean) | Precision (Mean) | Recall (Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Terpilih)** | **$0.8028; \pm 0.1114$** | **$0.5703$** | $0.7788$ | $0.5872$ | **$0.5628$** |
| **Random Forest** | $0.7753; \pm 0.0943$ | $0.5448$ | **$0.7951$** | **$0.6313$** | $0.4802$ |
| **XGBoost** | $0.7497; \pm 0.1689$ | $0.5358$ | $0.7894$ | $0.6275$ | $0.4682$ |
| **Support Vector Machine (RBF)** | $0.7270; \pm 0.1887$ | $0.5462$ | $0.7597$ | $0.6018$ | $0.5047$ |

### Alasan Pemilihan Model Terbaik (`Logistic Regression`)
1. **Generalisasi Spasial Tertinggi:** Mencapai nilai $\text{ROC-AUC} = 0.8028$, menunjukkan batas keputusan linear yang halus lebih tahan terhadap *overfitting* lokal pada blok wilayah baru dibandingkan model berbasis pohon.
2. **Sensitivitas Deteksi Bahaya:** Meraih nilai **Recall** ($0.5628$) dan **F1-Score** ($0.5703$) tertinggi sehingga paling efektif menekan risiko *False Negatives* (ruas jalan berbahaya yang terprediksi aman).
3. **Probabilitas Terkalibrasi & Ringan:** Menghasilkan indeks probabilitas kontinu ($0.0 - 1.0$) secara instan untuk klasifikasi 5 tingkat kerentanan pada antarmuka WebGIS.

---

## 6. Struktur Repositori dan Instalasi

### Struktur Direktori
```text
westjava-road-landslide-ml/
├── app.py                              # API FastAPI dan logika prediksi/pemrosesan spasial
├── data/
│   ├── dataset_landslide_corridor_clean.csv
│   └── shapefile_igt_2021_jalan_daerah_v2/
│       └── data_1793852851_7e866142.{shp,shx,dbf,prj}
├── docs/
│   ├── privacy.txt                     # Dokumen kebijakan privasi
│   └── toc.txt                         # Dokumen ketentuan penggunaan
├── models/
│   ├── landslide_model_lr.pkl          # Model Logistic Regression
│   ├── landslide_model_rf.pkl          # Model Random Forest
│   ├── landslide_model_svm.pkl         # Model Support Vector Machine
│   ├── landslide_model_xgb.json        # Model XGBoost
│   └── spatial_scaler.pkl              # StandardScaler untuk prapemrosesan
├── static/
│   ├── index.html                      # Antarmuka WebGIS
│   ├── favicon.ico
│   └── logo.webp
├── img/
│   └── logo.webp
├── Dockerfile                          # Image aplikasi Python
├── docker-compose.yaml                 # Orkestrasi layanan WebGIS
├── requirements.txt                    # Dependensi lingkungan lokal
├── requirements-docker.txt             # Dependensi image Docker
├── .dockerignore
└── README.md
```

### Panduan Instalasi & Menjalankan Aplikasi

#### Prasyarat

- Python `3.11` (disarankan)
- Pip dan `venv`
- Docker Desktop atau Docker Engine + Docker Compose *(opsional untuk instalasi Docker)*

#### Instalasi Lokal

1. **Kloning repositori**
   ```bash
   git clone https://github.com/ryanfaiz/westjava-road-landslide-ml.git
   cd westjava-road-landslide-ml
   ```

2. **Buat virtual environment dan instal dependensi**

   **Windows (PowerShell):**
   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   **Linux atau macOS:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

   Dependensi utama meliputi FastAPI, Uvicorn, Pandas, NumPy, GeoPandas, SciPy, Scikit-learn, XGBoost, Joblib, Shapely, dan Pydantic.

3. **Jalankan aplikasi**
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```
   Buka `http://127.0.0.1:8000`. Dokumentasi API otomatis tersedia di `http://127.0.0.1:8000/docs`.

Untuk menjalankan aplikasi dengan satu perintah tanpa menyiapkan environment Python:

### Menjalankan dengan Docker Compose

```bash
docker compose up --build
```

Aplikasi tersedia di `http://localhost:8000`. Untuk menghentikan dan menghapus container:

```bash
docker compose down
```

---

## 7. Keterbatasan dan Pengembangan Lanjutan

* **Resolusi Data & Variabel Geologi:** Dataset saat ini menggunakan DEM resolusi $90\text{ m}$ dan label statis ZKGT. Pengembangan selanjutnya akan mengintegrasikan **DEMNAS** ($8\text{ m}$), peta litologi batuan, jenis tanah, serta indeks vegetasi (**NDVI**).
* **Optimasi Ambang Klasifikasi:** Penyesuaian *classification threshold* dan skema *class weighting* untuk meningkatkan nilai *Recall* pada skenario curah hujan ekstrem.
* **Integrasi Data Dinamis:** Menghubungkan aplikasi WebGIS dengan API prakiraan curah hujan harian **BMKG** untuk mendukung sistem peringatan dini (*early warning system*) operasional.

---

## 8. Referensi Utama

1. **Haris, M., et al. (2026).** *Landslide susceptibility and road exposure modeling along linear transportation corridors using machine learning.*
2. **He, Q., et al. (2025).** *Spatial comparison of Random Forest and Support Vector Machine architectures for regional landslide susceptibility mapping.*
3. **Liang, Z., et al. (2025).** *Evaluating spatial stability of Random Forest models across varying topographic regimes.*
4. **Feng, L., et al. (2026).** *Capturing non-linear hydro-geomorphological interactions in landslide prediction using XGBoost and SVM.*
5. **Xie, W., et al. (2026).** *Gradient boosting optimization for complex landslide risk assessment balancing precision and recall.*

---

## 9. Tim Pengembang

**Kelas A — Kelompok 7 | Teknologi Rekayasa Komputer, Sekolah Vokasi IPB University**

| Nama Lengkap | NIM | Peran & Kontribusi Utama |
| :--- | :---: | :--- |
| **Ratu Festiza Alya Ichsan** | `J0404241174` | **Data Engineering & EDA Spasial:** Akuisisi data PUPR, DEM, CHIRPS, & ZKGT; reproyeksi UTM 48S; ekstraksi *buffer* $500\text{ m}$; visualisasi EDA & uji VIF. |
| **Ryan Faiz Sanie** | `J0404241078` | **Modeling & Validasi Spasial:** *Feature engineering* trigonometri; implementasi *5-Fold Spatial Block CV* (*KMeans*); pelatihan dan evaluasi 4 model ML. |
| **Dzaky Naufal Putra Raditio** | `J0404241020` | **WebGIS Deployment & Integrasi:** Serialisasi pipeline `.pkl`/`.json`; pengembangan antarmuka dan API WebGIS **FastAPI + HTML/JavaScript**; kalkulasi paparan segmen ($\text{km}$). |