<p align="center">
  <a href="README.md">English</a> |
  <a href="i18n/README_zh.md">中文</a> |
  <a href="i18n/README_hi.md">हिंदी</a> |
  <a href="i18n/README_es.md">Español</a> |
  <a href="i18n/README_fr.md">Français</a> |
  <a href="i18n/README_ar.md">العربية</a> |
  <a href="i18n/README_bn.md">বাংলা</a> |
  <a href="i18n/README_ru.md">Русский</a> |
  <a href="i18n/README_pt.md">Português</a> |
  <a href="i18n/README_id.md">Bahasa Indonesia</a>
</p>

<p align="center">
  <img src="assets/banner.png" alt="Banner Sistem Trading AI Hibrid" width="100%"/>
</p>

<div align="center">
  <br />
  <h1>📈 Sistem Trading AI Hibrid 📈</h1>
  <p>
    Sebuah sistem pakar pendukung keputusan untuk trading ETF NASDAQ dan Minyak (WTI), memanfaatkan kecerdasan buatan hibrid 12-model untuk sinyal trading yang andal dan bernuansa.
  </p>
</div>

<div align="center">

[![Status Proyek](https://img.shields.io/badge/status-dalam--pengembangan-green.svg)](https://github.com/laurentvv/Trading-AI)
[![Versi Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Lisensi](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📚 Daftar Isi

- [🌟 Tentang Proyek](#-tentang-proyek)
  - [🚀 Strategi Dual-Ticker (Analisis vs. Trading)](#-strategi-dual-ticker-analisis-vs-trading)
  - [🧠 Mesin AI Hibrid](#-mesin-ai-hibrid)
  - [🧘 Filosofi Keputusan: "Kehati-hatian Kognitif"](#-filosofi-keputusan-kehati-hatian-kognitif)
  - [✨ Fitur Utama](#-fitur-utama)
  - [💻 Stack Teknologi](#-stack-teknologi)
  - [⚙️ Performa & Perangkat Keras](#️-performa--perangkat-keras)
  - [🧠 Arsitektur AI & LLM (Gemini + Fallback Lokal)](#-arsitektur-ai--llm-gemini--fallback-lokal)
  - [🧠 FinAcumen (Memori Finansial)](#-finacumen-memori-finansial)
- [📂 Struktur Proyek](#-struktur-proyek)
- [🚀 Mulai Cepat](#-mulai-cepat)
  - [✅ Prasyarat](#-prasyarat)
  - [⚙️ Instalasi](#️-instalasi)
- [🛠️ Penggunaan](#️-penggunaan)
  - [Mode Simulasi (Paper Trading)](#mode-simulasi-paper-trading)
  - [Eksekusi Nyata (Trading 212)](#eksekusi-nyata-trading-212)
- [🧪 Backtesting Produksi](#-backtesting-produksi)
  - [Fitur](#fitur)
  - [Penggunaan](#penggunaan)
- [🤝 Berkontribusi](#-berkontribusi)
- [📜 Lisensi](#-lisensi)
- [📧 Kontak](#-kontak)

---

## 🌟 Tentang Proyek

Sebuah sistem pakar pendukung keputusan untuk trading ETF NASDAQ dan Minyak (WTI), memanfaatkan kecerdasan buatan hibrid 12-model.

### 🚀 Strategi Dual-Ticker (Analisis vs. Trading)

Sistem **menganalisis indeks global** (mis. `^NDX` untuk Nasdaq-100, `CL=F` untuk WTI) namun **mengeksekusi pada ETF denominasi EUR** (mis. `SXRV.DE`, `CRUDP.PA`). Pemisahan ini memastikan analisis pada data berketelitian tinggi dan eksekusi nyata pada aset yang dapat diakses melalui Trading 212.

### 🧠 Mesin AI Hibrid

Mesin menggabungkan model-model heterogen menjadi sebuah **konsensus berbobot**:

1. **Model Scikit-Learn** (RandomForest, GradientBoosting, LogisticRegression) — divalidasi dengan `TimeSeriesSplit` untuk mencegah kebocoran data. Sinyal kuantitatif agresif (25% dari bobot kognitif).
2. **TimesFM 2.5** (Google Research) — model fondasi untuk peramalan deret waktu.
3. **TensorTrade / PPO** (stable-baselines3) — agen Reinforcement Learning di lingkungan Gymnasium kustom.
4. **Gemma 4 12B** (Ollama) — analisis **tekstual** (makro/berita) dan **visual** (chart teknikal); **pertahanan JSON dua lapis** menjamin JSON yang bersih meski mode berpikir `<|think|>` aktif.
5. **Analisis Sentimen Hibrid** (Alpha Vantage + AlphaEar + Hyperliquid).
6. **Vincent Ganne Model** — kunci geopolitik (WTI, Brent, gas, urea, DXY) yang hanya menghasilkan sinyal BUY untuk memvalidasi dasar Nasdaq.
7. **OilBenchModel** — model kognitif khusus untuk WTI (indikator teknikal + fundamental EIA + sentimen).

### 🧘 Filosofi Keputusan: "Kehati-hatian Kognitif"

Model kognitif (Gemma 4, sentimen, Vincent Ganne) memegang **75%** bobot keputusan versus **25%** untuk model kuantitatif agresif. Pembobotan berlebih yang disengaja ini memastikan konteks kualitatif menperangkan sinyal kuantitatif. Sinyal hanya dieksekusi jika kepercayaan global melampaui **40%**; antara 20%–40% diturunkan menjadi HOLD.

### ✨ Fitur Utama

- **Arsitektur LLM Hibrid Cloud/Local**: integrasi `free-llm-api-keys` untuk memanfaatkan "Frontier Models" yang sangat cerdas (DeepSeek, Claude, Gemini) untuk analisis tekstual, dengan fallback lokal Ollama yang 100% andal (yang tetap menjadi mesin eksklusif untuk chart visual).
- **Pendekatan Dual-Ticker**: analisis indeks, trading ETF.
- **Harga live T212**: pengambilan harga EUR real-time melalui API Trading 212 (0,2 detik), dengan fallback yfinance dan cache parquet.
- **Spread Dated Brent**: pemantauan ketegangan pasar fisik melalui spread antara Brent Spot (Dated) dan Brent berjangka.
- **Ketahanan jaringan**: pemutus arus yfinance dengan pelacak terpisah (info vs. download), timeout 10 detik pada semua panggilan jaringan.
- **Invalidasi cache otomatis**: cache Parquet mendeteksi kedaluwarsa (> 2 hari) dan memaksa penyegaran. Gunakan `refresh_cache.py` untuk pembersihan manual.
- **Paralelisasi panggilan LLM**: panggilan model independen (`text_llm`, `visual_llm`, `search_query`, `timesfm`, `tensortrade`, `grebenkov`) berjalan dalam `ThreadPoolExecutor` untuk menumpuk inferensi Ollama dengan I/O. Jalur kritis biasanya 4–6 menit pada CPU dibanding 10+ menit berurutan.
- **Cache kueri-pencarian 24 jam**: kueri pencarian web yang dihasilkan LLM di-cache di `data_cache/search_queries/<ticker>_<date>_<price-sig>.json`. Kunci berdasarkan tanggal + tanda aksi-harga (pengelompokan log2 dari close + bucket RSI), sehingga perubahan regime membatalkannya. Kueri fallback **tidak pernah** di-cache (satu kegagalan Ollama transien tidak dapat meracuni cache selama 24 jam).
- **Timeout siklus ketat**: setiap siklus per ticker dibungkus dalam anggaran 40 menit (`CYCLE_TIMEOUT_SECONDS` di `main.py`). Saat timeout, thread pekerja di-`shutdown(wait=F)` agar ticker berikutnya segera dimulai; HOLD diterapkan ke ticker yang kedaluwarsa. Setiap future memiliki timeout per-tugas sendiri (pencarian 240 dtk, visual 300 dtk, teks 240 dtk, model CPU masing-masing 180 dtk, berita 90 dtk, crawl web 30 dtk).
- **Keamanan orphan-thread**: saat timeout siklus, sebuah `threading.Event` per-ticker disetel agar pekerja orphan keluar sebelum panggilan `execute_t212_trade` apa pun — mencegah perdagangan uang nyata setelah pengguna melihat panel "HOLD diterapkan". Sebuah `threading.Lock` per-ticker lebih lanjut menyerikan penempatan order T212, menghilangkan risiko perdagangan-ganda di bawah tumpang-tindih scheduler atau pemanggilan `--ticker` duplikat.
- **Sentinel kegagalan LLM**: ketika `_query_ollama` menghabiskan semua percobaan ulangnya, dict fallback membawa penanda `"failed": True` agar logika konsensus downstream dapat membedakan "model memilih HOLD" dari "model crash" (saat ini disebarkan namun tidak difilter — tindak lanjut yang diketahui).
- **Kognisi lanjutan**: penggunaan **Gemma 4 12B** dengan **pertahanan JSON dua lapis**:
  1. **Penegakan skema sisi-server** (`format: SCHEMA_*` dengan `additionalProperties: false`) — lapisan penopang; dilewatkan via parameter `format` Ollama di setiap titik panggilan. Skema didefinisikan di `src/llm_client.py` (`SCHEMA_TRADING_DECISION`, `SCHEMA_SEARCH_QUERY`, `SCHEMA_OIL_ALLOCATION`).
  2. **Akhiran system prompt defensif** (`"...never add a 'thought' key."`) — garis kedua yang redundan-tapi-tak-merugikan, dipertahankan sebagai belt-and-braces terhadap setiap regresi masa depan lapisan skema.

  Token pemikiran `<|think|>` **aktif** di keempat system prompt produksi (diaktifkan kembali 2026-06-06 di `main` setelah validasi pada cabang `think-mode`). Lapisan skema-lah yang benar-benar menetralkan cacat historis puing `<|channel>thought` JSON (penyebab akar Mei 2026): `tests/check_llm_json.py` mengonfirmasi bahwa kasus schema-strict (`v3_schema`, `v6_schema`, `v7_schema_strict`) menghasilkan JSON bersih bahkan dengan `<|think|>` aktif, sementara varian loose `format:json` gagal. Lihat `docs/ADR-001-think-mode-dual-layer-defence.md` untuk analisis lengkap dan prosedur pembalikan.
- **Agen Morning Brief otonom**: sebuah alur kerja semalam berbasis `smolagents` (`morning_brief/morning_brief.py`) yang dijadwalkan otomatis pada 01:00 via `schedule.py`. Ia secara independen merayapi log API harian, mengunduh data inventaris fundamental EIA, dan menjadi arbiter perdebatan *Bull vs Bear*. Laporan markdown yang dihasilkan (`morning_market_brief.md`) secara otomatis disuntikkan ke dalam system prompt LLM tekstual selama siklus trading harian, menganugerahkan AI utama memori kontekstual yang dalam dan kesadaran fundamental tanpa memperlambat eksekusi pasar live.
- **🏛️ Weekend Council (Memori Strategis)**: retrospektif LLM multi-persona mingguan (`src/council/weekend_council.py`) yang berjalan setiap **Sabtu pukul 01:00** via `schedule.py`. Enam persona — masing-masing pada **keluarga model Ollama yang berbeda** (Gemma 4 12B / GLM-4.6V-Flash / Qwen 3.5 9B / LFM 2.5 / Mistral Nemo 12B) untuk keberagaman penalaran yang nyata — berembuk dalam protokol 4 ronde (Problem Restate Gate → Analysis dengan STANCE eksplisit → Debat 1-vs-1 → sintesis Judge) dengan mekanisme anti-groupthink (kuota ketidaksetujuan, putusan unresolved-first). Judge (Qwen3.5-9B-MTP) memancarkan sikap per ticker yang menjadi **suara berbobot ke-11** (9,5%) dalam konsensus real-time, dengan kepercayaan yang meluruh secara linier selama 7 hari. Anggaran token yang murah hati (`num_predict` hingga 12000, `num_ctx` hingga 65536) dan jendela scheduler 48 jam mengakomodasi model berpikir pada CPU. Dewan menganalisis data PROD nyata: akurasi model (`model_performance.db`), metrik portofolio dan peringatan kritis (`performance_monitor.db`), serta jurnal trading yang tereksekusi. Instal 6 model yang diperlukan dengan `uv run python setup_council_models.py`. Lihat `docs/ADR-003-weekend-council-11th-voice.md`.
- **Berita & Sentimen Blockchain**: integrasi **AlphaEar** dan **Hyperliquid** untuk menangkap sentimen sosial dan spekulatif.
- **Penjadwal otomatis**: skrip `schedule.py` untuk eksekusi berkelanjutan (8:30 – 18:00) di server.
- **Manajemen risiko terpusat**: `AdvancedRiskManager` memusatkan logika Anti-Loss (Stop-Loss) dan Trailing Stop. Model individual tidak lagi mengelola risiko ini, memastikan strategi perlindungan modal yang terpadu dan ketat lintas regime pasar.
- **Kontrak data ketat**: semua model AI sepenuhnya terstandarisasi untuk mengembalikan dataclass `ModelResult` yang bertipe kuat (`signal`, `confidence`, `reasoning`), memastikan 100% keseragaman di seluruh mesin konsensus.
- **Kesehatan kode diaudit**: proyek mempertahankan standar kesehatan kode **Grade B** melalui audit otomatis (0 kode mati, indeks pemeliharaan tinggi).
- **Backtesting produksi**: mesin backtest mandiri (`backtest_prod.py`) yang memutar ulang sinyal prod nyata terhadap harga nyata dengan biaya T212 — tanpa dependensi eksternal.
- **Kontrol dump debug**: setel `TRADING_DEBUG_DUMP=0` untuk menonaktifkan dump (dibatasi 5 MB) `data_cache/llm_debug_fail.txt` atas kegagalan LLM.

### 💻 Stack Teknologi

- **Bahasa**: `Python 3.12+`
- **Perhitungan & Data**: `pandas`, `numpy`, `yfinance`, `pyarrow`, `pandas_datareader`, `hyperliquid-python-sdk`
- **Machine Learning**: `scikit-learn`, `shap`
- **AI & LLM**: `google-genai` (Gemini), `requests`, `ollama`
- **Scraping Web & Pencarian**: `beautifulsoup4`, `duckduckgo_search`, `crawl4ai`
- **Visualisasi**: `matplotlib` (backend Agg untuk thread safety), `seaborn`, `mplfinance`
- **Utilitas**: `tqdm`, `rich`, `python-dotenv`, `schedule`

### ⚙️ Performa & Perangkat Keras
Sistem ini dirancang untuk **berperforma pada perangkat keras konsumen** tanpa memerlukan GPU khusus.
- **Hanya CPU**: inferensi LLM (Gemma 4 12B Q6_K via Ollama) dan TimesFM berjalan sepenuhnya pada CPU. Throughput ~3–4 token/detik pada CPU 8-inti modern.
- **RAM yang disarankan**: minimum 16 GB (32 GB disarankan untuk menjalankan Gemma 4 12B dengan nyaman bersama TimesFM dan TensorTrade).
- **Konkurensi Ollama**: setel `OLLAMA_NUM_PARALLEL=8` (sudah di `.env` yang direkomendasikan) agar beberapa panggilan LLM berbagi beban model. Dengan anggaran konteks default 4 GB, slot paralel mendapat masing-masing ~512 token — Ollama akan menyerikan jika prompt melebihi ctx per-slot, namun `ThreadPoolExecutor` menjaga tumpang-tindih wall-clock yang bermanfaat untuk langkah-langkah yang terikat-I/O (pengambilan berita, crawl web, model CPU).
- **Waktu eksekusi**: ~6 sampai 9 menit per ticker pada CPU (dingin), ~3 sampai 5 menit per ticker saat cache kueri-pencarian terkena. Default menjalankan dua ticker (CRUDP.PA + SXRV.DE), jadi rencanakan ~15 menit total.
- **Timeout siklus**: setiap siklus per ticker dibatasi 40 menit (`CYCLE_TIMEOUT_SECONDS`). Jika terlampaui, HOLD diterapkan dan ticker berikutnya segera dimulai.
- **Kecepatan API**: integrasi Trading 212 ultra-cepat (<1 detik untuk pengambilan harga live).

### 🧠 Arsitektur AI & LLM (Gemini + Fallback Lokal)
Sistem memanfaatkan arsitektur multi-tingkat yang sangat andal untuk memastikan uptime maksimum dan pengambilan keputusan cerdas, yang terintegrasi mendalam ke dalam `main.py` dan Weekend Council.

- **Fallback Berjenjang 4-Tingkat**:
  1. **Tingkat Gemini Berbayar (`GEMINI_API_KEY_PAY`)**: Prioritas tertinggi. Menggunakan model lanjutan seperti Gemini 2.5 Pro untuk penalaran kompleks, visi chart teknikal, dan keputusan trading final.
  2. **Tingkat Gemini Gratis (`GEMINI_API_KEY`)**: Digunakan untuk tugas yang lebih ringan dan bervolume tinggi seperti peringkasan konteks web.
  3. **Proxy API LLM gratis**: Cadangan via `free-llm-api-keys`.
  4. **Ollama Lokal**: Fallback CPU offline 100% andal jika semua layanan cloud gagal.
- **Proteksi biaya**: tingkat berbayar dibatasi oleh anggaran biaya 30-hari bergulir (`GEMINI_PAY_MONTHLY_BUDGET_EUR`, default 8,6 €/bulan) — biaya setiap panggilan dihitung dari penggunaan token aktual × harga model dan diakumulasi; saat anggaran tercapai, panggilan jatuh ke tingkat gratis / Ollama. Backstop harian (`GEMINI_PAY_DAILY_CAP`, default 200) menjaga dari loop tak terkendali.
- **Integrasi**: mesin eksekusi harian utama (`main.py`) menggunakan Gemini untuk konsensus multi-model real-time, sementara Weekend Council asinkron (`council`) mengintegrasikan Gemini secara khusus untuk peran tertentu (seperti Judge dan Sceptique) berdampingan dengan beragam model Ollama lokal.

### 🧠 FinAcumen (Memori Finansial)
Arsitektur FinAcumen telah diintegrasikan untuk menganugerahkan model AI lokal sebuah **memori pengalaman** dan alat deterministik. Ini memecahkan masalah amnesia LLM.
- FinAcumen bekerja **secara asinkron pada malam hari** (via `schedule.py`) untuk memanfaatkan kekuatan penuh CPU tanpa memblokir siklus trading.
- Laporannya yang kualitatif dan mendalam otomatis ditambahkan ke **Morning Market Brief** untuk membimbing LLM keputusan sepanjang hari trading.

## 📂 Struktur Proyek

Proyek diorganisir secara modular untuk pemeliharaan yang lebih baik.

```
Trading-AI/
├── morning_brief/                   # Agen otonom semalam untuk analisis fundamental mendalam
│   ├── morning_brief.py             # Orkestrator agen dan konfigurasi smolagents
│   └── output/                      # Laporan markdown harian yang dihasilkan (morning_market_brief.md)
├── src/                             # Modul inti
│   ├── adaptive_weight_manager.py   # Pembobotan model dinamis berbasis performa
│   ├── advanced_risk_manager.py     # Manajemen risiko Trend-Aware dan sizing
│   ├── bootstrap.py                 # Logika inisialisasi inti
│   ├── chart_generator.py           # Menghasilkan chart teknikal untuk LLM visual
│   ├── classic_model.py             # Ensemble model kuantitatif Scikit-learn
│   ├── config_weights.py            # Konfigurasi bobot dasar mesin hibrid
│   ├── data.py                      # Pengambilan, cache, dan prapemrosesan data
│   ├── database.py                  # Manajemen database SQLite untuk metrik
│   ├── eia_client.py                # Klien API Energy Information Administration
│   ├── enhanced_decision_engine.py  # Mesin fusi hibrid yang mengorkestrasi semua model
│   ├── enhanced_trading_example.py  # Skrip contoh penggunaan model
│   ├── features.py                  # Rekayasa fitur teknikal dan makroekonomi
│   ├── grebenkov_model.py           # Model matematika Trend-Following (Agnostic Risk Parity)
│   ├── hmm_model.py                 # Hidden Markov Model untuk deteksi regime
│   ├── llm_client.py                # Integrasi Ollama untuk inferensi LLM lokal
│   ├── news_fetcher.py              # Perayapan dan parsing berita finansial
│   ├── oil_bench_model.py           # Model trading WTI khusus energi
│   ├── performance_monitor.py       # Pelacakan akurasi dan riwayat model
│   ├── read_simul.py                # Alat untuk membaca output simulasi
│   ├── sentiment_analysis.py        # Integrasi sentimen Alpha Vantage & AlphaEar
│   ├── t212_executor.py             # Eksekusi nyata API Trading 212 dan portofolio
│   ├── tensortrade_model.py         # Sinyal Reinforcement Learning (PPO)
│   ├── timesfm_model.py             # Integrasi peramalan deret waktu TimesFM 2.5
│   └── web_researcher.py            # Scraping web makroekonomi dengan Crawl4AI
├── data_cache/                       # Semua cache (di-gitignore)
│   ├── *.parquet                     # Data OHLCV per ticker (yfinance)
│   ├── macro/                        # Deret waktu makro (FRED, multi-sumber)
│   ├── search_queries/               # Cache 24 jam kueri-pencarian LLM (per ticker+tanggal+price-sig)
│   └── llm_debug_fail.txt            # Dump (dibatasi 5 MB) kegagalan LLM — nonaktifkan dengan TRADING_DEBUG_DUMP=0
├── tests/                            # Skrip pengujian dan validasi
│   ├── test_full_cycle.py            # Test end-to-end T212 beli/tunggu/jual
│   ├── test_enhanced_decision_engine.py # Test untuk mesin fusi hibrid
│   ├── check_llm_json.py             # Diagnostik JSON-schema LLM (menguji keempat titik panggilan Ollama)
│   ├── check_live.py                 # Skrip verifikasi harga pasar live
│   └── ...                           # Test unit dan integrasi lainnya
├── i18n/                            # Internasionalisasi (README yang diterjemahkan)
├── assets/                          # Aset statis (gambar, banner)
├── memory-bank/                     # State deterministik 4-file + konteks long-form (lihat AGENTS.md §1)
├── backtest_prod.py                 # Mesin backtest produksi mandiri
├── main.py                          # Satu-satunya titik masuk (Analisis & Trading)
├── pyproject.toml                   # Dependensi dan konfigurasi proyek (uv)
├── refresh_cache.py                 # Utilitas CLI untuk memaksa penyegaran cache Parquet
├── schedule.py                      # Penjadwal live untuk eksekusi otomatis
├── setup_timesfm.py                 # Skrip instalasi TimesFM 2.5 vendor
├── .env.example                     # Contoh variabel lingkungan
└── README.md                        # Dokumentasi ini
```

---

## 🚀 Mulai Cepat

Ikuti langkah-langkah berikut untuk menyiapkan lingkungan pengembangan lokal Anda.

### ✅ Prasyarat

- Python 3.12+ (via `uv`)
- [Ollama](https://ollama.com/) terpasang dan berjalan secara lokal.
- Model LLM yang telah diunduh: `ollama pull hf.co/unsloth/gemma-4-12b-it-GGUF:Q6_K`
- **Model Weekend Council** (opsional, namun diperlukan untuk keragaman penalaran dewan): dewan menjalankan setiap persona pada *keluarga* model yang *berbeda* (Gemma / GLM / Qwen / LFM). Pasang semuanya sekaligus dengan `uv run python setup_council_models.py`.

### ⚙️ Instalasi

1.  **Kloning repositori:**
    ```sh
    git clone https://github.com/laurentvv/Trading-AI.git
    cd Trading-AI
    ```
2.  **Pasang `uv` (jika belum):**
    Lihat [astral.sh/uv](https://astral.sh/uv) untuk instruksi pemasangan.

3.  **Buat dan aktifkan lingkungan virtual (LANGKAH KRITIS):**
    Anda harus membuat dan mengaktifkan `.venv` sebelum memasang model fondasi.
    ```bash
    uv venv
    source .venv/bin/activate  # Di Windows, gunakan `.\.venv\Scripts\activate.ps1`
    ```

4.  **Pasang Model Fondasi:**
    Jalankan skrip instalasi untuk mengkloning model ke `vendor/` dan menerapkan tambalan:
    ```bash
    python setup_timesfm.py
    ```

5.  **Inisialisasi dan sinkronisasi lingkungan:**
    ```bash
    uv sync
    ```

6.  **Pasang browser untuk riset Web (Crawl4AI):**
    ```bash
    uv run python -m playwright install chromium
    ```

7.  **Konfigurasikan kunci API Anda:**
    Buat berkas `.env` di root proyek:
    ```
    ALPHA_VANTAGE_API_KEY="KUNCI_ANDA"
    EIA_API_KEY="KUNCI_ANDA"

    # Opsional namun sangat disarankan: Integrasi Gemini AI
    GEMINI_API_KEY_PAY="KUNCI_TINGKAT_BERBAYAR_ANDA"  # Untuk penalaran/visi kompleks (Gemini 2.5 Pro)
    GEMINI_API_KEY="KUNCI_TINGKAT_GRATIS_ANDA"         # Untuk tugas yang lebih ringan (peringkasan)
    GEMINI_PAY_MONTHLY_BUDGET_EUR=8.6        # Anggaran biaya 30-hari bergulir (€) — penjaga penagihan utama
    GEMINI_PAY_DAILY_CAP=200                 # Backstop: maks. panggilan API berbayar per hari
    ```

---

## 🛠️ Penggunaan

Sistem melatih model-modelnya pada data terbaru di setiap eksekusi sebelum memberikan keputusan.

### Mode Simulasi (Paper Trading)

Untuk menguji sistem tanpa risiko dengan modal fiktif sebesar €1000, gunakan flag `--simul`. Sistem akan mengelola riwayat pembelian dan penjualan yang ketat.

```sh
# Jalankan analisis simulasi (Default: SXRV.DE - Nasdaq 100 EUR)
uv run main.py --simul

# Jalankan pada Minyak (WTI)
uv run main.py --ticker CRUDP.PA --simul
```

### Eksekusi Nyata (Trading 212)

Sistem kini **sepenuhnya terintegrasi** dengan Trading 212:
- **Verifikasi portofolio**: sebelum aksi apa pun, robot mengonsultasikan kas dan posisi nyata Anda.
- **Manajemen API**: mencakup mekanisme coba-ulang otomatis terhadap batas permintaan (Rate Limiting).

```sh
# Jalankan analisis dengan eksekusi nyata (Demo atau Real sesuai .env)
uv run main.py --t212
```

---

## 🧪 Backtesting Produksi

Sistem mencakup **mesin backtest produksi mandiri** (`backtest_prod.py`) yang memutar ulang sinyal prod nyata dari `logs_prod/trading_journal.csv` terhadap harga nyata dari berkas Parquet `data_cache/`.

### Fitur
- **Sinyal nyata**: memutar ulang keputusan tepat dari mesin hibrid 12-model.
- **Harga nyata**: menggunakan data OHLCV ETF aktual (SXRV.DE, CRUDP.PA) — tanpa proksi US.
- **Biaya T212**: mensimulasikan model biaya per-trade 0,1% Trading 212.
- **Perbandingan baseline**: secara otomatis menghitung performa buy-and-hold sebagai tolok ukur.
- **Metrik**: Rasio Sharpe, Drawdown Maksimum, Win Rate, Alpha, Total Imbal Hasil per ticker.

### Penggunaan

```bash
uv run python backtest_prod.py
```

Hasil disimpan ke `logs_prod/backtest_report.json` dengan kurva ekuitas dalam CSV.

---

## 🤝 Berkontribusi

Kontribusi diterima! Jangan ragu untuk men-fork proyek dan membuka Pull Request.

---

## 📜 Lisensi

Didistribusikan di bawah Lisensi MIT.

---

## 📧 Kontak

Tautan proyek: [https://github.com/laurentvv/Trading-AI](https://github.com/laurentvv/Trading-AI)
