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
  <img src="../assets/banner.png" alt="Hybrid AI Trading Banner" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Hybrid AI Trading System 📈</h1>
  <p>
    Sistem pendukung keputusan pakar untuk trading ETF NASDAQ dan Minyak (WTI), memanfaatkan kecerdasan buatan hibrida tri-modal untuk sinyal trading yang kuat dan bernuansa.
  </p>
</div>

<div align="center">

[![Project Status](https://img.shields.io/badge/status-in--development-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Daftar Isi

- [🌟 Tentang Proyek](#-tentang-proyek)
  - [✨ Fitur Utama](#-fitur-utama)
  - [💻 Tech Stack](#-tech-stack)
  - [⚙️ Performa & Perangkat Keras](#️-performa--perangkat-keras)
- [📂 Struktur Proyek](#-struktur-proyek)
- [🚀 Mulai Cepat](#-mulai-cepat)
  - [✅ Prasyarat](#-prasyarat)
  - [⚙️ Instalasi](#️-instalasi)
- [🛠️ Penggunaan](#️-penggunaan)
  - [Mode Simulasi (Paper Trading)](#mode-simulasi-paper-trading)
  - [Eksekusi Nyata (Trading 212)](#eksekusi-nyata-trading-212)
- [🧪 Backtesting Produksi](#-backtesting-produksi)
- [🤝 Kontribusi](#-kontribusi)
- [📜 Lisensi](#-lisensi)
- [📧 Kontak](#-kontak)

---

## 🌟 Tentang Proyek

Proyek ini adalah sistem pendukung keputusan pakar untuk trading ETF, menggunakan pendekatan AI hibrida tri-modal. Sistem ini dirancang untuk memberikan analisis komprehensif dan kuat dengan menggabungkan beberapa perspektif AI.

### 🚀 Strategi Dual-Ticker (Analisis vs. Trading)
Sistem menggunakan pendekatan inovatif untuk memaksimalkan akurasi model:
- **Analisis Akurasi Tinggi**: Model AI menganalisis **indeks referensi global** (`^NDX` untuk Nasdaq, `CL=F` untuk Minyak Mentah WTI). Indeks ini menawarkan riwayat yang lebih panjang dan tren yang lebih "murni", tanpa kebisingan terkait jam trading atau biaya ETF.
- **Eksekusi ETF**: Pesanan nyata ditempatkan pada ticker yang sesuai di **Trading 212** (`SXRV.DE`, `CRUDP.PA`), menggunakan **harga langsung T212** (via API posisi) untuk menentukan ukuran posisi. Status portofolio disinkronkan langsung dari T212 (`sync_state_from_t212()`), dan harga langsung dimasukkan ke dalam alur kerja analisis (`_inject_t212_live_price()` di `src/data.py`).

### 🧠 Mesin AI Hibrida
Sistem menggabungkan sebelas sinyal berbeda:
1.  **Model Kuantitatif Klasik**: Ansambel RandomForest/GradientBoosting/LogisticRegression yang dilatih dengan indikator teknikal dan makroekonomi.
2.  **TimesFM 2.5 (Google Research)**: Model fondasi mutakhir untuk peramalan deret waktu.
3.  **TensorTrade / PPO (Reinforcement Learning)**: Agen RL (stable-baselines3) yang melatih kebijakan PPO dalam lingkungan trading Gymnasium kustom dengan persistensi antar siklus.
4.  **Model Oil-Bench (Gemma 4 12B (Unsloth))**: Model khusus energi yang menggabungkan data fundamental **EIA** (Stok, Impor, Utilisasi Kilang) dan sentimen untuk trading WTI.
5.  **LLM Tekstual (Gemma 4 12B (Unsloth))**: Analisis kontekstual data mentah, berita real-time via skill **AlphaEar**, dan integrasi **riset web makro-ekonomi** dinamis. Model ini secara eksplisit membaca laporan **Morning Brief** semalaman untuk mendapatkan kesadaran fundamental yang mendalam sebelum membuat keputusan.
6.  **LLM Visual (Gemma 4 12B (Unsloth))**: Analisis langsung pada grafik teknikal (`enhanced_trading_chart.png`).
7.  **Analisis Sentimen**: Analisis hibrida yang menggabungkan Alpha Vantage dan tren "panas" dari **AlphaEar** (Weibo, WallstreetCN).
8.  **Data Terdesentralisasi (Hyperliquid)**: Analisis sentimen spekulatif pada Minyak (WTI) melalui *Funding Rate* dan *Open Interest*.
9.  **Model Vincent Ganne**: Analisis geopolitik dan lintas aset (WTI, Brent, Gas, DXY, MA200) untuk mendeteksi titik terendah makroekonomi.
10. **Model Grebenkov**: Model matematis Mengikuti Tren yang dikalibrasi untuk analisis lintas aset menggunakan Paritas Risiko Agnostik.
11. **Mesin Fusi Hibrida**: Meta-model yang mengorkestrasi pembobotan dinamis dan konsensus kognitif di antara semua sub-model.

Tujuannya adalah menghasilkan keputusan akhir (`BUY`, `SELL`, `HOLD`) dengan prioritas mutlak pada **Akurasi Pertama**.

### 🧘 Filosofi Keputusan: "Kehati-hatian Kognitif"
Berbeda dengan algoritme trading klasik yang panik begitu volatilitas melonjak, sistem ini menerapkan pendekatan investor yang terinformasi:
- **Diperlukan Konsensus Kuat**: Model kuantitatif (Klasik) mungkin menyuarakan bahaya (`SELL`), tetapi jika model kognitif (LLM Teks, Visual, TimesFM) tetap netral, sistem akan lebih memilih `HOLD`.
- **Filter Kepercayaan**: Keputusan pergerakan (Beli atau Jual) hanya divalidasi jika kepercayaan global melampaui ambang batas keamanan (umumnya 40%). Di bawah ini, sistem menganggap sinyal sebagai "kebisingan" dan tetap bersiaga.
- **Perlindungan Modal**: Dalam mode risiko `VERY_HIGH`, `HOLD` berfungsi sebagai perisai. Ini mencegah masuk ke pasar yang tidak stabil dan menghindari keluar terlalu cepat pada koreksi teknis sederhana jika fundamental (Berita/Visual/Hyperliquid) tidak mengkonfirmasi jatuhnya pasar dalam waktu dekat.

### ✨ Fitur Utama

- **Pendekatan Dual-Ticker**: Menganalisis indeks, trading pada ETF.
- **Harga Langsung T212**: Pemulihan harga EUR secara real-time melalui API Trading 212 (0,2 dtk), dengan cadangan yfinance dan cache parquet.
- **Spread Brent Tertanggal**: Pemantauan ketegangan pasar fisik melalui spread antara Brent Spot (Dated) dan Brent Futures.
- **Ketahanan Jaringan**: Pemutus arus yfinance dengan pelacak terpisah (info vs unduhan), batas waktu 10 dtk pada semua panggilan jaringan.
- **Auto-Invalidasi Cache**: Cache parquet secara otomatis mendeteksi data usang (> 2 hari) dan memaksa pembaruan. Gunakan `refresh_cache.py` untuk membersihkan cache secara manual.
- **Paralelisasi Panggilan LLM**: Panggilan model independen (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) berjalan dalam `ThreadPoolExecutor` untuk menumpuk inferensi Ollama dengan I/O. Jalur kritis biasanya memakan waktu 4–6 menit pada CPU vs 10+ menit secara sekuensial.
- **Cache Pencarian-Query 24 Jam**: Kueri pencarian web yang dihasilkan oleh LLM di-cache di bawah `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Diberi kunci berdasarkan tanggal + tanda tangan aksi harga (log2 bucketing dari penutupan + bucket RSI), sehingga perubahan rezim akan membatalkannya. Kueri cadangan **tidak pernah** di-cache (satu kegagalan sementara Ollama tidak akan meracuni cache selama 24 jam).
- **Batas Waktu Siklus Keras**: Setiap siklus ticker dibatasi oleh anggaran 15 menit (`CYCLE_TIMEOUT_SECONDS` di `main.py`). Jika melampaui batas waktu, thread pekerja akan di-`shutdown(wait=False)` sehingga ticker berikutnya segera dimulai; HOLD diterapkan pada ticker yang kehabisan waktu. Masing-masing *future* memiliki batas waktunya sendiri per tugas (pencarian 240 dtk, visual 300 dtk, teks 240 dtk, model CPU 180 dtk masing-masing, berita 90 dtk, perayapan web 30 dtk).
- **Keamanan Utas Yatim (Orphan-Thread Safety)**: Saat batas waktu siklus tercapai, `threading.Event` per-ticker ditetapkan sehingga pekerja yang yatim keluar sebelum panggilan `execute_t212_trade` apa pun — mencegah perdagangan uang sungguhan setelah pengguna ditunjukkan panel "HOLD appliqué". Sebuah `threading.Lock` per-ticker lebih lanjut menserialisasi penempatan pesanan T212, menghilangkan risiko perdagangan ganda di bawah tumpang tindih penjadwal atau pemanggilan ganda `--ticker`.
- **Penjaga Kegagalan LLM (LLM Failure Sentinel)**: Ketika `_query_ollama` menghabiskan semua percobaan ulang, kamus cadangan membawa bendera `"failed": True` sehingga logika konsensus hilir dapat membedakan "model memilih HOLD" dari "model mogok" (saat ini disebarkan tetapi tidak disaring — sebuah tindak lanjut yang diketahui).
- **Kognisi Lanjutan**: Penggunaan **Gemma 4 12B** dengan **pertahanan JSON lapis ganda**:
  1. **Penerapan skema sisi server** (`format: SCHEMA_*` dengan `additionalProperties: false`) — lapisan penahan beban utama; diteruskan melalui parameter `format` Ollama di setiap lokasi panggilan. Skema didefinisikan di `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Akhiran prompt sistem defensif** (`"...never add a 'thought' key."`) — garis pertahanan kedua yang redundan-tapi-tidak-berbahaya, dipertahankan sebagai pengaman ganda terhadap regresi lapisan skema di masa mendatang.

  Token penalaran `<|think|>` diaktifkan (**active**) di keempat prompt sistem produksi (diaktifkan kembali pada 2026-06-06 di `main` setelah validasi di cabang `think-mode`). Lapisan skema inilah yang benar-benar menetralisir kecacatan historis residu JSON `<|channel>thought` (akar penyebab pada Mei 2026): `tests/check_llm_json.py` mengonfirmasi bahwa kasus skema ketat (`v3_schema`, `v6_schema`, `v7_schema_strict`) menghasilkan JSON yang bersih bahkan dengan `<|think|>` diaktifkan, sementara varian `format:json` longgar gagal. Lihat `docs/ADR-001-think-mode-dual-layer-defence.md` untuk analisis lengkap dan prosedur pengembalian.
- **Agen Otonom Morning Brief**: Sebuah proses semalaman berbasis `smolagents` (`morning_brief/morning_brief.py`) yang dijadwalkan berjalan otomatis pada 01:00 AM via `schedule.py`. Agen ini secara independen menganalisis log API harian, mengunduh data fundamental inventaris EIA, dan mengarbitrase debat *Bull vs Bear*. Laporan markdown yang dihasilkan (`morning_market_brief.md`) secara otomatis disuntikkan ke dalam prompt sistem LLM Tekstual selama siklus trading harian, memberikan AI utama memori kontekstual dan kesadaran fundamental yang mendalam tanpa memperlambat eksekusi di pasar langsung.
- **Sentimen Berita & Blockchain**: Integrasi **AlphaEar** dan **Hyperliquid** untuk menangkap sentimen sosial dan spekulatif.
- **Penjadwal Otomatis**: Skrip `schedule.py` untuk eksekusi berkelanjutan (08:30 - 18:00) pada server.
- **Manajemen Risiko Terpusat**: `AdvancedRiskManager` memusatkan logika Anti-Loss (Stop-Loss) dan Trailing Stop. Model individu tidak lagi mengelola risiko ini, memastikan strategi perlindungan modal yang terpadu dan ketat di berbagai rezim pasar.
- **Kontrak Data yang Ketat**: Semua model AI sepenuhnya distandarisasi untuk mengembalikan dataclass `ModelResult` yang strongly-typed (`signal`, `confidence`, `reasoning`), memastikan keseragaman 100% di seluruh mesin konsensus.
- **Kesehatan Kode Diaudit**: Proyek mempertahankan standar kesehatan kode **Nilai B** melalui audit otomatis (0 dead code, indeks pemeliharaan tinggi).
- **Backtesting Produksi**: Mesin backtest mandiri (`backtest_prod.py`) yang memutar ulang sinyal produksi nyata terhadap harga nyata dengan biaya T212 — tanpa ketergantungan eksternal.
- **Kontrol Pembuangan Debug**: Atur `TRADING_DEBUG_DUMP=0` untuk menonaktifkan pembuangan (dump) kegagalan LLM yang dibatasi (5 MB) `data_cache/llm_debug_fail.txt`.

### 💻 Tech Stack

- **Bahasa**: `Python 3.12+`
- **Kalkulasi & Data**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **AI & LLM**: `requests`, `ollama`
- **Web Scraping & Pencarian**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualisasi**: `matplotlib` (Backend Agg untuk keamanan thread), `seaborn`, `mplfinance`
- **Utilitas**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performa & Perangkat Keras
Sistem ini dirancang agar **berperforma tinggi pada perangkat keras konsumen** tanpa memerlukan GPU khusus.
- **Hanya CPU**: Inferensi LLM (Gemma 4 12B Q6_K via Ollama) dan TimesFM berjalan sepenuhnya pada CPU. Throughput adalah sekitar 3–4 token/dtk pada CPU 8-inti modern.
- **RAM Direkomendasikan**: Minimal 16 GB (disarankan 32 GB untuk menjalankan Gemma 4 12B dengan nyaman berdampingan dengan TimesFM dan TensorTrade).
- **Konkurensi Ollama**: Atur `OLLAMA_NUM_PARALLEL=8` (sudah disarankan di `.env`) sehingga panggilan LLM ganda dapat berbagi beban model. Dengan anggaran konteks default 4 GB, slot paralel masing-masing mendapatkan ~512 token — Ollama akan menserialisasi jika prompt melampaui konteks per slot, tetapi `ThreadPoolExecutor` menjaga tumpang tindih waktu dinding tetap bermanfaat untuk langkah-langkah yang terikat I/O (pengambilan berita, perayapan web, model CPU).
- **Waktu Eksekusi**: ~6 hingga 9 menit per ticker pada CPU (keadaan dingin), ~3 hingga 5 menit per ticker saat mengenai cache kueri pencarian. Standar eksekusi menjalankan dua ticker (CRUDP.PA + SXRV.DE), jadi perkirakan total ~15 menit.
- **Batas Waktu Siklus**: Setiap siklus ticker dibatasi pada 15 menit (`CYCLE_TIMEOUT_SECONDS`). Jika terlampaui, HOLD akan diterapkan dan ticker berikutnya akan langsung dimulai.
- **Kecepatan API**: Integrasi Trading 212 ultra-cepat (<1 dtk untuk pemulihan harga langsung).

---

## 📂 Struktur Proyek

Proyek ini disusun secara modular untuk pemeliharaan yang lebih baik.

```
Trading-AI/
├── morning_brief/                   # Agen otonom semalaman untuk analisis fundamental mendalam
│   ├── morning_brief.py             # Orkestrator agen dan konfigurasi smolagents
│   └── output/                      # Laporan markdown yang dihasilkan harian (morning_market_brief.md)
├── src/                             # Modul inti
│   ├── adaptive_weight_manager.py   # Pembobotan model dinamis berdasarkan performa
│   ├── advanced_risk_manager.py     # Manajemen risiko dan sizing yang menyadari tren
│   ├── chart_generator.py           # Menghasilkan grafik teknikal untuk LLM visual
│   ├── classic_model.py             # Ansambel model kuantitatif Scikit-learn
│   ├── data.py                      # Pengambilan data, caching, dan prapemrosesan
│   ├── database.py                  # Manajemen basis data SQLite untuk metrik
│   ├── eia_client.py                # Klien API Energy Information Administration
│   ├── enhanced_decision_engine.py  # Mesin fusi hibrida yang mengorkestrasi semua model
│   ├── features.py                  # Rekayasa fitur teknikal dan makroekonomi
│   ├── grebenkov_model.py           # Model matematis pengikut tren (Agnostic Risk Parity)
│   ├── llm_client.py                # Integrasi Ollama untuk inferensi LLM lokal
│   ├── news_fetcher.py              # Pengambilan dan penguraian berita keuangan
│   ├── oil_bench_model.py           # Model trading WTI khusus energi
│   ├── performance_monitor.py       # Pelacakan akurasi model dan riwayat
│   ├── sentiment_analysis.py        # Integrasi sentimen Alpha Vantage & AlphaEar
│   ├── t212_executor.py             # Eksekusi nyata API Trading 212 dan portofolio
│   ├── tensortrade_model.py         # Sinyal Reinforcement Learning (PPO)
│   ├── timesfm_model.py             # Integrasi peramalan deret waktu TimesFM 2.5
│   └── web_researcher.py            # Web scraping makro-ekonomi dengan Crawl4AI
├── data_cache/                       # Semua cache (diabaikan oleh git)
│   ├── *.parquet                     # Data OHLCV per ticker (yfinance)
│   ├── macro/                        # Deret waktu makro (FRED, multi-sumber)
│   ├── search_queries/               # Cache kueri pencarian LLM 24 jam (per ticker+tanggal+tanda-harga)
│   └── llm_debug_fail.txt            # Pembuangan kegagalan LLM dibatasi (5 MB) — nonaktifkan via TRADING_DEBUG_DUMP=0
├── tests/                            # Skrip pengujian dan validasi
│   ├── test_full_cycle.py            # Tes ujung-ke-ujung beli/tunggu/jual T212
│   ├── test_enhanced_decision_engine.py # Tes untuk mesin fusi hibrida
│   ├── check_llm_json.py             # Diagnostik skema-JSON LLM (menguji ke-4 lokasi panggilan Ollama)
│   ├── check_live.py                 # Skrip verifikasi harga pasar langsung
│   └── ...                           # Tes unit dan integrasi lainnya
├── i18n/                            # Internasionalisasi (README Terjemahan)
├── assets/                          # Aset statis (gambar, spanduk)
├── memory-bank/                     # Memori asisten AI dan konteks
├── backtest_prod.py                 # Mesin backtest produksi mandiri
├── main.py                          # Titik masuk tunggal (Analisis & Trading)
├── pyproject.toml                   # Konfigurasi dan dependensi proyek (uv)
├── refresh_cache.py                 # Utilitas CLI untuk memaksa pembaruan cache Parquet
├── schedule.py                      # Penjadwal langsung untuk eksekusi otomatis
├── setup_timesfm.py                 # Skrip instalasi untuk vendor TimesFM 2.5
├── .env.example                     # Contoh variabel lingkungan
└── README.md                        # Dokumentasi ini
```

---

## 🚀 Mulai Cepat

Ikuti langkah-langkah ini untuk menyiapkan lingkungan pengembangan lokal Anda.

### ✅ Prasyarat

- Python 3.12+ (melalui `uv`)
- [Ollama](https://ollama.com/) terinstal dan berjalan secara lokal.
- Mengunduh model LLM: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`

### ⚙️ Instalasi

1.  **Klon repositori:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Instal `uv` (jika belum dilakukan):**
    Lihat [astral.sh/uv](https://astral.sh/uv) untuk instruksi instalasi.

3.  **Buat dan aktifkan virtual environment (Langkah KRUSIAL):**
    Anda harus membuat dan mengaktifkan `.venv` sebelum menginstal model fondasi.
    ```bash
    uv venv
    source .venv/bin/activate  # Di Windows, gunakan `.\.venv\Scripts\activate.ps1`
    ```

4.  **Instal Model Fondasi:**
    Jalankan skrip instalasi untuk mengkloning model ke dalam `vendor/` dan menerapkan tambalan (patch):
    ```bash
    python setup_timesfm.py
    ```

5.  **Inisialisasi dan sinkronkan lingkungan:**
    ```bash
    uv sync
    ```

6.  **Instal browser untuk riset Web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **Konfigurasikan kunci API Anda:**
    Buat file `.env` di akar proyek:
    ```
    ALPHA_VANTAGE_API_KEY="KUNCI_ANDA"
    EIA_API_KEY="KUNCI_ANDA"
    ```

---

## 🛠️ Penggunaan

Sistem melatih model-modelnya pada data terbaru di setiap eksekusi sebelum memberikan keputusan.

### Mode Simulasi (Paper Trading)

Untuk menguji sistem tanpa risiko dengan modal fiktif sebesar €1000, gunakan bendera `--simul`. Sistem akan mengelola riwayat beli dan jual yang ketat.

```sh
# Jalankan analisis simulasi (Default: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Jalankan pada Minyak (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Eksekusi Nyata (Trading 212)

Sistem ini sekarang **sepenuhnya terintegrasi** dengan Trading 212:
- **Verifikasi Portofolio**: Sebelum mengambil tindakan apa pun, robot berkonsultasi dengan uang tunai dan posisi nyata Anda.
- **Manajemen API**: Termasuk mekanisme coba lagi (retry) otomatis terhadap batasan permintaan (Rate Limiting).

```sh
# Jalankan analisis dengan eksekusi nyata (Demo atau Real berdasarkan .env)
uv run main.py --t212
```

---

## 🧪 Backtesting Produksi

Sistem ini mencakup **mesin backtest produksi mandiri** (`backtest_prod.py`) yang memutar ulang sinyal produksi sebenarnya dari `logs_prod/trading_journal.csv` terhadap harga nyata dari file Parquet `data_cache/`.

### Fitur
- **Sinyal nyata**: Memutar ulang keputusan pasti dari mesin hibrida 12-model.
- **Harga nyata**: Menggunakan data OHLCV ETF aktual (SXRV.DE, CRUDP.PA) — bukan proksi AS.
- **Biaya T212**: Mensimulasikan model biaya per transaksi Trading 212 sebesar 0,1%.
- **Perbandingan dasar**: Secara otomatis menghitung performa beli-dan-tahan (buy-and-hold) sebagai patokan.
- **Metrik**: Rasio Sharpe, Penarikan Maksimum (Maximum Drawdown), Tingkat Kemenangan (Win Rate), Alpha, Total Imbal Hasil (Total Return) per ticker.

### Penggunaan

```bash
uv run python backtest_prod.py
```

Hasil disimpan ke `logs_prod/backtest_report.json` beserta kurva ekuitas CSV.

---

## 🤝 Kontribusi

Kontribusi dipersilakan! Jangan ragu untuk me-fork proyek ini dan membuka Pull Request.

---

## 📜 Lisensi

Didistribusikan di bawah Lisensi MIT.

---

## 📧 Kontak

Tautan Proyek: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)
