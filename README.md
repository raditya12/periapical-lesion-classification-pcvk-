# Klasifikasi Lesion Periapikal — Proyek Akhir PCVK

Klasifikasi citra radiografi periapikal ke dalam dua kelas (**lesion** vs **healthy**) menggunakan
pengolahan citra klasik + machine learning: CLAHE, segmentasi Otsu, fitur GLCM/LBP, lalu SVM & Random Forest.

**Kelompok 1 · TI3E **
Mata Kuliah Pengolahan Citra dan Visi Komputer (PCVK)

| Anggota | Bagian |
|---|---|---|
| Destian Dwi Hardika  | Dataset & Praproses |
| Raditya Riefki  | Segmentasi & ROI |
| Indhira Yuantika Christy  | Ekstraksi Fitur |
| Fifa Nuurun Halizah  | Klasifikasi & Evaluasi |

---

## Dataset

**Lesion_Dataset** — Periapical Radiographs for Lesion Classification
- Sumber: Zenodo, DOI [10.5281/zenodo.18824283](https://doi.org/10.5281/zenodo.18824283)
- Lisensi: CC BY 4.0
- Isi: 2.285 citra grayscale TIFF — 1.192 lesion / 1.093 healthy — 1.295 pasien
- Dataset **tidak disertakan** di repo ini (ukuran ±470 MB). Unduh dari Zenodo, lalu letakkan di
  Google Drive dengan struktur:

```
MyDrive/
└── Lesion_Dataset/
    ├── lesion/          (1.192 file .TIF)
    └── healthy_root/    (1.093 file .TIF)
```

---

## Struktur Repo

```
.
├── notebooks/
│   ├── 00_Verifikasi_Dataset.ipynb            # cek dataset + screenshot jumlah citra per kelas
│   ├── 01_Proyek_Akhir_PCVK.ipynb             # MASTER: pipeline lengkap (Run all)
│   ├── 01a_Dataset_Praproses.ipynb            # Destian  — EDA, split, CLAHE, analisis praproses
│   ├── 01b_Raditya_Segmentasi_ROI.ipynb       # Raditya  — Otsu, morfologi, apex, ROI
│   ├── 01c_Indhira_Ekstraksi_Fitur.ipynb      # Indhira  — GLCM, LBP, bentuk, analisis fitur
│   └── 01d_Fifa_Klasifikasi_Evaluasi.ipynb    # Fifa     — SVM, RF, evaluasi, perbandingan
├── docs/
│   ├── 01_Rencana_Proyek_Akhir_PCVK.pdf
│   ├── 02_Penjelasan_Metode_Proyek.pdf
│   └── 03_Penjelasan_Sel_Notebook.pdf
├── .gitignore
└── README.md
```

---

## Pipeline

```
Citra radiografi periapikal (.TIF)
   │
   ├─ Praproses     : Grayscale → Resize 512×320 → CLAHE (clipLimit 2.0, tile 8×8) → Gaussian 5×5
   ├─ Segmentasi/ROI: Otsu thresholding → closing(6) → opening(3) → area gigi → apex → crop ROI
   ├─ Ekstraksi     : GLCM (4 fitur, jarak [1,3], 4 sudut) + LBP (nri_uniform, 59 bin) + bentuk (4)
   ├─ Klasifikasi   : SVM (kernel RBF) vs Random Forest (500 pohon)
   └─ Evaluasi      : Accuracy, Sensitivity, Specificity, F1, AUC, Confusion Matrix
```

**Pembagian data per pasien** (GroupShuffleSplit 70/15/15) — bukan per citra, karena 114 pasien
memiliki citra di kedua kelas sehingga split per citra akan menyebabkan *data leakage*.

---

## Hasil

Diuji pada **345 citra test** yang belum pernah dilihat model:

| Konfigurasi | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|
| **CLAHE + Whole + LBP + SVM** ⭐ | **73,6%** | **79,6%** | 67,1% | 76,0% | **0,815** |
| CLAHE + Whole + GLCM+LBP+bentuk + SVM | 72,2% | 75,7% | 68,3% | 74,1% | 0,801 |
| Tanpa CLAHE + Whole + gabungan + SVM | 71,0% | 81,2% | 59,8% | 74,6% | 0,769 |
| CLAHE + ROI + gabungan + SVM | 67,5% | 82,9% | 50,6% | 72,8% | 0,727 |
| GLCM saja (SVM) | 55,1% | 56,4% | 53,7% | 56,8% | 0,567 |

**Temuan utama**
1. **CLAHE bermanfaat** (+1,2 poin accuracy dibanding tanpa CLAHE).
2. **ROI otomatis di sekitar apex justru lebih rendah** dari whole-image — crop membuang konteks global,
   dan sebagian apex meleset pada gigi yang terpotong tepi citra.
3. **LBP (59 fitur) lebih kuat dari GLCM (4 fitur)** pada dataset ini; gabungan keduanya tidak selalu menang.
4. **SVM konsisten mengalahkan Random Forest** di 8 dari 8 pasangan skenario.

---

## Cara Menjalankan

1. Upload dataset ke Drive (lihat bagian Dataset di atas).
2. Buka notebook di Google Colab — pilih **Runtime → Change runtime type → CPU** (tidak perlu GPU).
3. Jalankan `notebooks/01_Proyek_Akhir_PCVK.ipynb` → **Runtime → Run all**.
4. Fitur hasil ekstraksi otomatis di-cache ke `MyDrive/proyek_pcvk_cache/` sehingga run berikutnya
   hanya butuh ±15 detik (bukan 7 menit).

---

## Catatan Teknis

- **LBP `nri_uniform`, bukan `uniform`.** Dengan `P=8`, `method='uniform'` di scikit-image hanya
  menghasilkan 10 nilai (0–9); jika di-histogram ke 59 bin maka **49 bin selalu nol**. Mode yang benar
  untuk 59 bin adalah `nri_uniform` — setelah diperbaiki tidak ada fitur mati dan akurasi naik
  (70,5% → 75,0% pada uji subset).
- **Cache fitur berversi** (`CACHE_VERSION = 'v3'`). Bila cara ekstraksi berubah, naikkan versi agar
  cache lama otomatis diabaikan.
- **Pemilihan model terbaik memakai AUC**, bukan sensitivity saja — sensitivity dapat digelembungkan
  dengan lebih sering menebak "lesion" (specificity ikut jatuh).
- **SMOTE tidak dipakai.** Referensi A4 (Peng dkk., 2025) menunjukkan SMOTE justru menurunkan AUC
  pada data uji (0,544 vs 0,758; p = 0,018).

---

## Referensi

1. Z. Can and E. Aydin, "Explainable CNN–Radiomics Fusion and Ensemble Learning for Multimodal Lesion
   Classification in Dental Radiographs," *Diagnostics*, vol. 15, no. 16, art. 1997, 2025.
   doi: 10.3390/diagnostics15161997
2. V. S. Kumar et al., "Machine learning in the detection of dental cyst, tumor, and abscess lesions,"
   *BMC Oral Health*, vol. 23, art. 833, 2023. doi: 10.1186/s12903-023-03571-1
3. R. Widyaningrum, E. I. Sela, R. Pulungan, and A. Septiarini, "Automatic Segmentation of Periapical
   Radiograph Using Color Histogram and Machine Learning for Osteoporosis Detection,"
   *International Journal of Dentistry*, vol. 2023, art. 6662911, 2023. doi: 10.1155/2023/6662911
4. M. Peng, B. Yu, J. Hu, X. Xie, and J. He, "Radiomics-based classification of pediatric dental trauma
   in periapical radiographs: a preliminary study," *BMC Medical Imaging*, vol. 25, art. 336, 2025.
   doi: 10.1186/s12880-025-01877-w
5. Z. Can, "Lesion_Dataset: Periapical Radiographs for Lesion Classification (Version 1.0) [Data set],"
   Zenodo, 2026. doi: 10.5281/zenodo.18824283

---

## Lisensi & Etika

Kode dalam repo ini bebas digunakan untuk keperluan akademik. Dataset mengikuti lisensi **CC BY 4.0**
dari Zenodo dan wajib disitasi. Sistem ini adalah **prototipe akademik / alat bantu skrining**,
bukan alat diagnosis medis — keputusan klinis tetap di tangan dokter gigi.
