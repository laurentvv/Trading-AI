<p align="center">
  <a href="../README.md">English</a> | 
  <a href="README_zh.md">中文</a> | 
  <a href="README_hi.md">हिंदी</a> | 
  <a href="README_es.md">Español</a> | 
  <a href="README_fr.md">Français</a> | 
  <a href="README_ar.md">العربية</a> | 
  <a href="README_bn.md">বাংলা</a> | 
  <a href="README_ru.md">Русский</a> | 
  <a href="README_pt.md">Português</a> | 
  <a href="README_id.md">Bahasa Indonesia</a>
</p>

<p align="center">
  <img src="../assets/banner.png" alt="Banner Trading AI Hybrid" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistem Trading AI Hybrid 📈</h1>
  <p>
    Sistem pendukung keputusan ahli untuk perdagangan ETF NASDAQ dan Minyak (WTI), memanfaatkan kecerdasan buatan hybrid tri-modal untuk sinyal perdagangan yang kuat dan bernuansa.
  </p>
</div>

<div align="center">

[![Status Proyek](https://img.shields.io/badge/status-dalam--pengembangan-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Versi Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lisensi](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  <img src="../enhanced_performance_dashboard.png" alt="Dashboard Performa" width="800"/>
</p>

---

## 📚 Daftar Isi

- [🌟 Tentang Proyek](#-tentang-proyek)
  - [✨ Fitur Utama](#-fitur-utama)
  - [💻 Stack Teknologi](#-stack-teknologi)
  - [⚙️ Performa & Perangkat Keras](#️-performa--perangkat-keras)
- [📂 Struktur Proyek](#-struktur-proyek)
- [🚀 Mulai Cepat](#-mulai-cepat)
  - [✅ Prasyarat](#-prasyarat)
  - [⚙️ Instalasi](#️-instalasi)
- [🛠️ Penggunaan](#️-penggunaan)
  - [Analisis Manual](#-analisis-manual)
  - [Analisis Otomatis dengan Penjadwal Cerdas](#-analisis-otomatis-dengan-penjadwal-cerdas)
- [🤝 Berkontribusi](#-berkontribusi)
- [📜 Lisensi](#-lisensi)
- [📧 Kontak](#-kontak)

---

## 🌟 Tentang Proyek

Proyek ini adalah sistem pendukung keputusan ahli untuk perdagangan ETF, menggunakan pendekatan AI hybrid tri-modal. Sistem ini dirancang untuk memberikan analisis yang komprehensif dan kuat dengan menggabungkan beberapa perspektif AI.

### 🚀 Strategi Dual-Ticker (Analisis vs Trading)
Sistem menggunakan pendekatan inovatif untuk memaksimalkan akurasi model:
- **Analisis High-Fidelity**: Model AI menganalisis **indeks referensi global** (`^NDX` untuk Nasdaq, `CL=F` untuk Minyak Mentah WTI). Indeks ini menawarkan riwayat yang lebih panjang dan tren yang "lebih murni", tanpa kebisingan terkait jam perdagangan atau biaya ETF.
- **Eksekusi ETF**: Pesanan nyata ditempatkan pada ticker yang sesuai di **Trading 212** (`SXRV.DE`, `CRUDP.PA`), menggunakan **harga live T212** (via API posisi) untuk penentuan ukuran posisi.

### 🧠 Mesin AI Hybrid
Sistem menggabungkan delapan sinyal berbeda:
1.  **Model Kuantitatif Klasik**: Ensemble RandomForest/GradientBoosting/LogisticRegression yang dilatih pada indikator teknis dan makroekonomi.
2.  **TimesFM 2.5 (Google Research)**: Model fondasi mutakhir untuk peramalan deret waktu (time-series).
3.  **Model Oil-Bench (Gemma 4:e4b)**: Model khusus energi yang menggabungkan data fundamental **EIA** (Stok, Impor, Utilisasi Kilang) dan sentimen untuk perdagangan WTI.
4.  **LLM Tekstual (Gemma 4:e4b)**: Analisis kontekstual dari data mentah, berita real-time melalui skill **AlphaEar**, dan integrasi **riset web makro-ekonomi** yang dinamis.
5.  **LLM Visual (Gemma 4:e4b)**: Analisis langsung grafik teknis (`enhanced_trading_chart.png`).
6.  **Analisis Sentimen**: Analisis hybrid yang menggabungkan Alpha Vantage dan tren "panas" dari **AlphaEar** (Weibo, WallstreetCN).
7.  **Data Terdesentralisasi (Hyperliquid)**: Analisis sentimen spekulatif pada Minyak (WTI) melalui *Funding Rate* dan *Open Interest*.
8.  **Model Vincent Ganne**: Analisis geopolitik dan lintas-aset (WTI, Brent, Gas, DXY, MA200) untuk mendeteksi titik terendah makroekonomi.

Tujuannya adalah untuk menghasilkan keputusan akhir (`BUY`, `SELL`, `HOLD`) dengan prioritas mutlak pada **Akurasi Utama**.

### 🧘 Filosofi Keputusan: "Kewaspadaan Kognitif"
Berbeda dengan algoritma perdagangan klasik yang panik segera setelah volatilitas meledak, sistem ini menerapkan pendekatan investor yang terinformasi:
- **Konsensus Kuat Diperlukan**: Model kuantitatif (Klasik) mungkin memberi sinyal jual (`SELL`), tetapi jika model kognitif (LLM Teks, Visi, TimesFM) tetap netral, sistem akan lebih memilih `HOLD`.
- **Filter Kepercayaan**: Keputusan pergerakan (Beli atau Jual) hanya divalidasi jika kepercayaan global melebihi ambang batas keamanan (umumnya 40%). Di bawah ini, sistem menganggap sinyal sebagai "kebisingan" dan tetap dalam posisi siaga.
- **Perlindungan Modal**: Dalam mode risiko `VERY_HIGH`, `HOLD` berfungsi sebagai perisai. Ini mencegah memasuki pasar yang tidak stabil dan menghindari keluar sebelum waktunya pada koreksi teknis sederhana jika fundamental (Berita/Visi/Hyperliquid) tidak mengonfirmasi kehancuran yang akan datang.

### ✨ Fitur Utama

- **Pendekatan Dual-Ticker**: Analisis indeks, perdagangkan ETF.
- **Harga Live T212**: Pemulihan harga EUR real-time melalui API Trading 212 (0,2 detik), dengan fallback yfinance dan cache parquet.
- **Dated Brent Spread**: Pemantauan ketegangan pasar fisik melalui selisih antara Brent Spot (Dated) dan Brent Futures.
- **Ketahanan Jaringan**: Pemutus arus (circuit breaker) yfinance dengan pelacak terpisah (info vs download), timeout 10 detik pada semua panggilan jaringan.
- **Kognisi Tingkat Lanjut**: Penggunaan **Gemma 4** untuk sintesis teknis/fundamental yang lebih baik.
- **Sentimen Berita & Blockchain**: Integrasi **AlphaEar** dan **Hyperliquid** untuk menangkap sentimen sosial dan spekulatif.
- **Penjadwal Otomatis**: Skrip `schedule.py` untuk eksekusi berkelanjutan (08.30 - 18.00) di server.
- **Manajemen Risiko Lanjutan**: Penyesuaian sinyal otomatis berdasarkan volatilitas dan rezim pasar.

### 💻 Stack Teknologi

- **Bahasa**: `Python 3.12+`
- **Perhitungan & Data**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **AI & LLM**: `requests`, `ollama`
- **Web Scraping & Pencarian**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualisasi**: `matplotlib`, `seaborn`, `mplfinance`
- **Utilitas**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performa & Perangkat Keras
Sistem dirancang untuk **berkinerja pada perangkat keras konsumen** tanpa memerlukan GPU khusus.
- **Hanya CPU**: Inferensi LLM (Gemma 4 via Ollama) dan TimesFM dioptimalkan untuk eksekusi CPU yang cepat jika RAM yang tersedia cukup.
- **RAM yang Direkomendasikan**: Minimum 16 GB (disarankan 32 GB untuk menjalankan Gemma 4 dengan nyaman).
- **Waktu Eksekusi**: ~2 hingga 5 menit untuk siklus penuh (termasuk web crawling, pelatihan ML, prediksi TimesFM, dan 3 analisis LLM).
- **Kecepatan API**: Integrasi Trading 212 ultra-cepat (<1 detik untuk pemulihan harga live).

---

## 📂 Struktur Proyek

Proyek diatur secara modular untuk pemeliharaan yang lebih baik.

```
Trading-AI/
├── src/                     # Modul inti
│   ├── eia_client.py               # Klien data fundamental energi
│   ├── oil_bench_model.py          # Model khusus energi
│   ├── enhanced_decision_engine.py # Mesin fusi dan model Vincent Ganne
│   ├── advanced_risk_manager.py    # Manajemen risiko sadar tren
│   ├── adaptive_weight_manager.py  # Manajemen bobot model dinamis
│   ├── t212_executor.py            # Eksekusi nyata di Trading 212
│   ├── timesfm_model.py            # Integrasi TimesFM 2.5
│   └── ...                         # Data, Fitur, Klien LLM
├── tests/                   # Skrip pengujian dan validasi
├── data_cache/              # Data pasar dan makro (Parquet)
├── main.py                  # Titik masuk tunggal (Analisis & Trading)
├── schedule.py              # Penjadwal langsung (08.30 - 18.00)
├── backtest_engine.py       # Mesin backtesting historis
├── .env                     # Kunci API (Alpha Vantage, T212, EIA)
└── README.md                # Dokumentasi ini
```

---

## 🚀 Mulai Cepat

Ikuti langkah-langkah berikut untuk menyiapkan lingkungan pengembangan lokal Anda.

### ✅ Prasyarat

- Python 3.12+ (melalui `uv`)
- [Ollama](https://ollama.com/) terinstal dan berjalan secara lokal.
- Model LLM yang telah diunduh: `ollama pull gemma4:e4b`

### ⚙️ Instalasi

1.  **Clone repositori:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instal `uv` (jika belum dilakukan):**
    Lihat [astral.sh/uv](https://astral.sh/uv) untuk instruksi instalasi.

3.  **Instal dan Patch TimesFM 2.5 (Langkah KRUSIAL):**
    Jalankan skrip instalasi untuk mengkloning model ke `vendor/` dan menerapkan patch:
    ```bash
    python setup_timesfm.py
    ```

4.  **Inisialisasi dan sinkronkan lingkungan:**
    ```bash
    uv sync
    ```

5.  **Instal browser untuk riset web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

6.  **Konfigurasikan kunci API Anda:**
    Buat file `.env` di root proyek:
    ```
    ALPHA_VANTAGE_API_KEY="KUNCI_ANDA"
    EIA_API_KEY="KUNCI_ANDA"
    ```

---

## 🛠️ Penggunaan

Sistem melatih modelnya pada data terbaru pada setiap eksekusi sebelum memberikan keputusan.

### Mode Simulasi (Paper Trading)

Untuk menguji sistem tanpa risiko dengan modal fiktif sebesar €1000, gunakan flag `--simul`. Sistem akan mengelola riwayat beli dan jual yang ketat.

```sh
# Jalankan analisis simulasi (Default: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Jalankan pada Minyak (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Eksekusi Nyata (Trading 212)

Sistem sekarang **terintegrasi penuh** dengan Trading 212:
- **Verifikasi Portofolio**: Sebelum tindakan apa pun, robot berkonsultasi dengan uang tunai dan posisi nyata Anda.
- **Manajemen API**: Termasuk mekanisme percobaan ulang otomatis terhadap batas permintaan (Rate Limiting).

```sh
# Jalankan analisis dengan eksekusi nyata (Demo atau Real sesuai .env)
uv run main.py --t212
```

---

## 🤝 Berkontribusi

Kontribusi sangat disambut! Jangan ragu untuk melakukan fork pada proyek dan membuka Pull Request.

---

## 📜 Lisensi

Didistribusikan di bawah Lisensi MIT.

---

## 📧 Kontak

Link Proyek: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)
