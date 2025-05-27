# Laporan Proyek Machine Learning Terapan - Reksi Hendra Pratama

## Domain Proyek: Telekomunikasi

## 1. Latar Belakang

Industri telekomunikasi merupakan salah satu sektor yang menghasilkan data dalam jumlah sangat besar setiap harinya. Salah satu tantangan utama yang dihadapi adalah **customer churn**, yaitu kondisi ketika pelanggan memutuskan untuk berhenti menggunakan layanan yang disediakan. Kehilangan pelanggan tidak hanya berdampak pada penurunan pendapatan, tetapi juga meningkatkan biaya akuisisi pelanggan baru.

Menurut beberapa studi, biaya untuk memperoleh pelanggan baru bisa mencapai lima kali lipat dibandingkan mempertahankan pelanggan lama, sehingga perusahaan harus fokus pada **strategi retensi pelanggan berbasis prediksi** [1], [2]. Dengan memanfaatkan teknik machine learning, data histori pelanggan dapat dianalisis untuk mengidentifikasi pelanggan yang berpotensi melakukan churn, sehingga memungkinkan perusahaan mengambil tindakan preventif secara lebih proaktif.

Penerapan model prediksi churn juga telah menunjukkan hasil yang menjanjikan. Misalnya, studi oleh Wagh et al. [2] menggunakan algoritma Random Forest dan mencapai akurasi hingga **99.09%**, menunjukkan potensi besar dari pendekatan ini dalam mendukung pengambilan keputusan bisnis yang strategis. Selain itu, prediksi churn membantu meningkatkan nilai umur pelanggan (Customer Lifetime Value) dan mempertahankan loyalitas pelanggan melalui layanan yang dipersonalisasi.

### Referensi
[1] Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). *Customer churn prediction in telecom using machine learning in big data platform*. Journal of Big Data, 6(28). https://doi.org/10.1186/s40537-019-0191-6  
[2] Wagh, S. K., et al. (2023). *Customer churn prediction in telecom sector using machine learning techniques*. Results in Control and Optimization, 14, 100342. https://doi.org/10.1016/j.rico.2023.100342


## 🎯 2. Business Understanding

### Problem Statement
- Bagaimana memprediksi pelanggan yang akan melakukan churn?
- Faktor-faktor apa saja yang paling berkontribusi terhadap kemungkinan pelanggan melakukan churn?

### Goals
- Mengembangkan model klasifikasi untuk mengidentifikasi pelanggan yang memiliki kemungkinan besar akan churn.
- Menggali fitur-fitur paling signifikan yang memengaruhi perilaku churn.
- Memberikan insight yang dapat digunakan tim marketing atau retensi untuk mengambil keputusan yang lebih baik.

### Solution Statement
- Membangun beberapa model seperti Logistic Regression, Random Forest, SVM, Naive Bayes.
- Menggunakan teknik visualisasi dan pemilihan fitur untuk menganalisis variabel penting.
- Evaluasi dilakukan berdasarkan metrik klasifikasi: accuracy, precision, recall, F1-score.


---

## 3. Data Understanding
### Informasi Dataset

| Keterangan       | Detail                                                                 |
|------------------|------------------------------------------------------------------------|
| Jumlah Data      | 7.043 baris                                                            |
| Jumlah Fitur     | 21 kolom                                                               |
| Target           | `Churn` (Yes / No)                                                     |
| Format           | CSV                                                                    |
| Sumber           | [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

### Contoh Data (10 Fitur)

| customerID | gender | SeniorCitizen | Partner | Dependents | tenure | PhoneService | InternetService | Contract     | PaymentMethod          | MonthlyCharges | TotalCharges | Churn |
|------------|--------|----------------|---------|------------|--------|---------------|------------------|--------------|------------------------|----------------|--------------|--------|
| 7590-VHVEG | Female | 0              | Yes     | No         | 1      | No            | DSL              | Month-to-month | Electronic check     | 29.85          | 29.85        | Yes    |
| 5575-GNVDE | Male   | 0              | No      | No         | 34     | Yes           | DSL              | One year       | Mailed check         | 56.95          | 1889.5       | No     |
| 3668-QPYBK | Male   | 0              | No      | No         | 2      | Yes           | Fiber optic      | Month-to-month | Electronic check     | 53.85          | 108.15       | Yes    |
| 7795-CFOCW | Male   | 0              | Yes     | No         | 45     | No            | No               | One year       | Bank transfer (auto) | 42.30          | 1840.75      | No     |
| 9237-HQITU | Female | 0              | Yes     | Yes        | 2      | Yes           | Fiber optic      | Month-to-month | Electronic check     | 70.70          | 151.65       | Yes    |


### Daftar Fitur dalam Dataset

| Nama Fitur         | Deskripsi                                                                 |
|--------------------|---------------------------------------------------------------------------|
| `customerID`        | ID unik untuk setiap pelanggan                                            |
| `gender`            | Jenis kelamin pelanggan (`Male` atau `Female`)                           |
| `SeniorCitizen`     | Status lansia (0 = bukan lansia, 1 = lansia)                             |
| `Partner`           | Status memiliki pasangan (`Yes` atau `No`)                               |
| `Dependents`        | Status memiliki tanggungan seperti anak atau orang tua (`Yes` atau `No`) |
| `tenure`            | Lama berlangganan dalam bulan                                            |
| `PhoneService`      | Apakah pelanggan menggunakan layanan telepon (`Yes` atau `No`)           |
| `MultipleLines`     | Apakah pelanggan menggunakan lebih dari satu jalur telepon               |
| `InternetService`   | Jenis layanan internet yang digunakan (`DSL`, `Fiber optic`, `No`)       |
| `OnlineSecurity`    | Layanan keamanan online (`Yes`, `No`, atau `No internet service`)        |
| `OnlineBackup`      | Layanan backup data online (`Yes`, `No`, atau `No internet service`)     |
| `DeviceProtection`  | Proteksi perangkat (`Yes`, `No`, atau `No internet service`)             |
| `TechSupport`       | Dukungan teknis (`Yes`, `No`, atau `No internet service`)                |
| `StreamingTV`       | Layanan streaming TV (`Yes`, `No`, atau `No internet service`)           |
| `StreamingMovies`   | Layanan streaming film (`Yes`, `No`, atau `No internet service`)         |
| `Contract`          | Jenis kontrak langganan (`Month-to-month`, `One year`, `Two year`)       |
| `PaperlessBilling`  | Apakah pelanggan menerima tagihan tanpa kertas (`Yes` atau `No`)         |
| `PaymentMethod`     | Metode pembayaran yang digunakan (misal: `Electronic check`)             |
| `MonthlyCharges`    | Total biaya yang dibayarkan setiap bulan oleh pelanggan                  |
| `TotalCharges`      | Total biaya yang dibayarkan selama menjadi pelanggan                     |
| `Churn`             | Target/label: Apakah pelanggan berhenti (`Yes`) atau tetap (`No`)        |

### Pemeriksaan Data Awal

#### Struktur Data
Dataset memiliki 21 kolom dengan 7043 baris data pelanggan. Semua baris memiliki nilai unik di kolom `customerID`.

#### Missing Values
Pemeriksaan awal menunjukkan tidak ada missing values pada sebagian besar kolom. Namun, kolom `TotalCharges` awalnya bertipe `object` karena berisi nilai kosong berupa spasi (`" "`). Setelah dikonversi ke tipe `float`, nilai kosong ini menjadi `NaN`. Sebanyak 11 nilai `NaN` ditemukan dan ditangani pada tahap data preparation.

#### Duplikasi Data
Tidak ditemukan duplikasi baris berdasarkan pemeriksaan pada data.

#### Tipe Data Tidak Konsisten
- `TotalCharges` perlu dikonversi dari `object` ke `float` agar bisa digunakan sebagai fitur numerik.
- Kolom target `Churn` bertipe string (`Yes`/`No`) dan perlu dikonversi ke numerik (`1`/`0`) sebelum modeling.

#### Outlier
Berdasarkan pemeriksaan outlier, data tampak normal

---

### 📉 Distribusi Churn Pelanggan

Gambar di bawah menunjukkan bahwa mayoritas pelanggan **tidak melakukan churn** (sekitar 73%), sementara hanya sekitar **27% pelanggan yang churn**.

![Screenshot 2025-05-25 144300](https://github.com/user-attachments/assets/437737ea-bbde-41da-9e88-ec99becb647d)


### 🔗 Korelasi Fitur Numerik

Heatmap berikut memperlihatkan korelasi antara fitur numerik terhadap variabel target `Churn`.

- `tenure` memiliki korelasi negatif dengan churn (-0.35), menunjukkan pelanggan yang lebih lama cenderung tidak churn.
- `MonthlyCharges` memiliki korelasi positif (0.19), menunjukkan semakin tinggi tagihan, kemungkinan churn meningkat.

![Screenshot 2025-05-25 144438](https://github.com/user-attachments/assets/6c955d97-82b0-4f57-96c1-56f0f28f7af5)



---

## 4. Data Preparation

### 4.1 Konversi dan Penanganan `TotalCharges`
- Kolom `TotalCharges` dikonversi ke `float`.
- Sebanyak 11 nilai kosong yang menjadi `NaN` diisi dengan **median** kolom tersebut.

### 4.2 Encoding Target
- Kolom target `Churn` dikonversi dari `'Yes'`/`'No'` menjadi `1`/`0`.

### 4.3 Penghapusan Kolom Tidak Relevan
- Kolom `customerID` dihapus karena tidak memiliki kontribusi prediktif.

### 4.4 Train-Test Split
- Data dibagi dengan rasio **70:30** menggunakan `train_test_split`.
- Hasil pembagian:
  - **Data latih**: 4930 baris
  - **Data uji**: 2113 baris
- Stratifikasi digunakan untuk memastikan distribusi churn tetap seimbang.

### 4.5 Encoding dan Normalisasi
- Fitur kategorikal diencode menggunakan **OneHotEncoder** (`drop='first'`).
- Fitur numerik seperti `tenure`, `MonthlyCharges`, dan `TotalCharges` dinormalisasi menggunakan **MinMaxScaler**.
- Hasil akhir encoding dan normalisasi digabung menjadi `X_train_final` dan `X_test_final` yang siap digunakan untuk pelatihan model.
"""

---

## 🤖 5. Modeling

### 5.1 Setup Model
DataFrame disiapkan untuk menyimpan hasil evaluasi dari beberapa algoritma klasifikasi yang akan digunakan, yaitu:

- **Bernoulli Naive Bayes**
- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

Model dipilih berdasarkan karakteristik data yang memiliki kombinasi fitur kategorikal dan numerik serta target biner (`Churn`).

---

### 5.2 Training dan Evaluasi Model

#### ✅ Bernoulli Naive Bayes
**Cara kerja:**  
Bernoulli Naive Bayes adalah algoritma klasifikasi berbasis **Teorema Bayes** yang mengasumsikan bahwa fitur-fitur bersifat **saling independen**. Model ini menghitung probabilitas kelas berdasarkan keberadaan atau ketidakhadiran fitur biner. Untuk setiap fitur, model memperkirakan probabilitas bahwa fitur tersebut bernilai 1 (ada) atau 0 (tidak ada) terhadap setiap kelas (`Churn = Yes` atau `No`), dan hasil akhirnya didapat dengan mengalikan semua probabilitas tersebut. Model ini sangat cocok untuk fitur hasil dari **One-Hot Encoding**.

Model ini menggunakan parameter default, dengan beberapa penyesuaian sebagai berikut:
- `alpha=1.0`: smoothing Laplace untuk menghindari probabilitas nol.
- `binarize=0.0`: fitur dibinarisasi berdasarkan threshold nol.
- `fit_prior=True`: menghitung probabilitas awal dari distribusi data pelatihan.
- `class_prior=None`: prior kelas tidak ditentukan secara manual.

- **Kelebihan:**
  - Cepat dalam pelatihan dan prediksi.
  - Cocok untuk data biner seperti hasil One-Hot Encoding.
  - Memiliki recall tinggi (baik untuk mendeteksi churn).
- **Kekurangan:**
  - Precision rendah (banyak false positive).
  - Asumsi independensi antar fitur sering tidak realistis.

---

#### 🌲 Random Forest Classifier

**Cara kerja:**  
Random Forest merupakan model **ensemble** yang terdiri dari banyak **decision trees**. Setiap pohon dibangun dari **subset acak** data (menggunakan teknik **bagging**) dan subset fitur. Saat melakukan prediksi, setiap pohon memberikan "suara" (voting), dan hasil akhir diambil dari mayoritas. Pendekatan ini membuat model tahan terhadap overfitting dan meningkatkan akurasi.

Model ini dikonfigurasi dengan:
- `n_estimators=100`: menggunakan 100 pohon keputusan.
- `random_state=42`: untuk reprodusibilitas hasil.

- **Kelebihan:**
  - Dapat menangani fitur kategorikal dan numerik dengan baik.
  - Cenderung tidak overfitting karena menggunakan banyak pohon.
  - Memberikan informasi tentang pentingnya fitur.
- **Kekurangan:**
  - Waktu pelatihan lebih lama.
  - Kurang transparan dibanding model linier.

---

#### 📈 Logistic Regression

**Cara kerja:**  
Logistic Regression memodelkan hubungan antara fitur dan target biner dengan menggunakan **fungsi sigmoid**. Model ini menghitung kombinasi linier dari input fitur, lalu memetakan hasilnya ke nilai antara 0 dan 1, yang diinterpretasikan sebagai probabilitas kelas positif (`Churn = Yes`). Jika probabilitas di atas threshold (biasanya 0.5), maka diklasifikasikan sebagai churn.

Model ini dikonfigurasi dengan:
- `max_iter=1000`: batas maksimum iterasi ditingkatkan untuk memastikan konvergensi.
- `random_state=42`: untuk memastikan hasil konsisten.

- **Kelebihan:**
  - Mudah diinterpretasikan dan transparan.
  - Cepat dalam pelatihan dan prediksi.
  - Seimbang dalam precision dan recall.
- **Kekurangan:**
  - Asumsi hubungan linear antara fitur dan target logit.
  - Kurang efektif untuk data yang sangat kompleks atau tidak linear.

---

#### 🔵 Support Vector Machine (SVM)
**Cara kerja:**  
SVM bekerja dengan mencari **hyperplane optimal** yang dapat memisahkan dua kelas dengan **margin maksimum**. Model ini hanya bergantung pada data yang berada paling dekat dengan batas pemisah (disebut **support vectors**). Untuk data yang tidak dapat dipisahkan secara linier, digunakan **kernel trick** (seperti RBF) untuk memetakan data ke ruang berdimensi lebih tinggi, sehingga dapat dipisahkan secara linier di ruang tersebut.


Model ini menggunakan:
- `kernel='rbf'`: kernel radial basis function untuk menangani non-linearitas.
- `probability=True`: memungkinkan estimasi probabilitas (berguna untuk evaluasi probabilistik).
- `random_state=42`: untuk memastikan hasil dapat direproduksi.

- **Kelebihan:**
  - Kuat untuk data berdimensi tinggi.
  - Dapat menangani non-linearitas dengan kernel (misalnya RBF).
- **Kekurangan:**
  - Tidak efisien untuk dataset besar.
  - Parameter C dan gamma perlu tuning hati-hati.

---


## 📈 6. Evaluation

### 🔢 Metrik Evaluasi yang Digunakan

| **Metrik**     | **Deskripsi** |
|----------------|----------------|
| **Accuracy**   | Rasio prediksi yang benar dari seluruh prediksi. Cocok jika data seimbang. |
| **Precision**  | Dari seluruh prediksi churn, berapa banyak yang benar-benar churn. Fokus untuk menghindari false positive. |
| **Recall**     | Dari seluruh pelanggan yang benar-benar churn, berapa banyak yang berhasil dideteksi model. Fokus untuk menghindari false negative. |
| **F1-Score**   | Rata-rata harmonis antara precision dan recall. Cocok jika perlu keseimbangan antara keduanya. |

> **Catatan**:
> - **Recall tinggi** penting jika perusahaan ingin memastikan semua pelanggan berisiko churn tertangkap (meskipun ada false positive).
> - **Precision tinggi** penting jika ingin menghindari salah target pelanggan yang tidak akan churn.
> - **F1-score** membantu menyeimbangkan keduanya.

---

### 📊 Hasil Evaluasi Masing-Masing Model

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 80.45%   | 65.81%    | 54.90% | 0.5986   |
| Random Forest         | 78.70%   | 62.59%    | 49.20% | 0.5509   |
| SVM                   | 79.32%   | 64.29%    | 49.73% | 0.5608   |
| Naive Bayes (BernoulliNB) | 70.89%   | 47.16%    | 79.86% | 0.5930   |


### 🔍 Visualisasi Confusion Matrix

Berikut adalah visualisasi confusion matrix untuk masing-masing model:

#### 📊 Bernoulli Naive Bayes
![Screenshot 2025-05-25 144453](https://github.com/user-attachments/assets/7eae5260-0ca8-4d3b-a488-5bac9eabf897)

#### 🌲 Random Forest
![Screenshot 2025-05-25 144516](https://github.com/user-attachments/assets/00f21ccb-ae8d-464f-82c4-330bfeabb19b)

#### 📈 Logistic Regression
![Screenshot 2025-05-25 144523](https://github.com/user-attachments/assets/bffa4bf3-9118-423b-b2cd-cef8d614368b)

#### 🔵 Support Vector Machine (SVM)
![Screenshot 2025-05-25 144530](https://github.com/user-attachments/assets/1f9b1cca-07ab-4d62-8cda-9e11292882e2)



---

## 📉 7. Analisis Hasil & Relevansi terhadap Business Understanding

Berdasarkan hasil evaluasi model, berikut adalah analisis lebih lanjut serta keterkaitannya dengan tujuan bisnis:

### 🔎 Analisis Model

- **Logistic Regression** menunjukkan performa terbaik secara keseluruhan dengan **F1-score tertinggi** (0.5986). Ini menunjukkan keseimbangan yang baik antara mendeteksi churn dan menghindari kesalahan prediksi.
- **Naive Bayes** memiliki **recall tertinggi** (79.86%), artinya model ini sangat sensitif dalam mendeteksi pelanggan yang akan churn, tetapi precision-nya rendah (banyak false positive).
- **Random Forest dan SVM** menawarkan performa yang seimbang, namun sedikit di bawah Logistic Regression dari sisi F1-score.

### 🧠 Relevansi Terhadap Business Understanding

Tujuan utama proyek ini adalah membantu perusahaan memprediksi pelanggan yang berisiko churn agar bisa diberikan tindakan pencegahan seperti:
- Penawaran spesial
- Penguatan layanan
- Komunikasi personal

Model yang **recall-nya tinggi** (seperti Naive Bayes) cocok digunakan jika perusahaan lebih mementingkan untuk **mendeteksi sebanyak mungkin churn**, meskipun berisiko false positive.

Namun, untuk pendekatan yang **lebih seimbang dan efisien**, **Logistic Regression** adalah pilihan terbaik:
- Tidak terlalu banyak false positive
- Tetap mampu menangkap sebagian besar churn
- Mudah diimplementasikan dan dijelaskan ke pihak non-teknis

### ✅ Rekomendasi Strategi
- Gunakan Logistic Regression sebagai model utama prediksi churn.
- Gunakan Naive Bayes sebagai second-opinion model dalam sistem early-warning churn dengan threshold yang lebih ketat.
- Fokuskan retensi pada pelanggan hasil prediksi yang memiliki profil risiko tinggi berdasarkan fitur seperti:
  - Tenure pendek
  - Metode pembayaran: electronic check
  - Jenis kontrak: bulanan

---
## 💡 8. Kesimpulan

Proyek ini berhasil membangun model machine learning untuk memprediksi risiko churn pada pelanggan perusahaan telekomunikasi dengan performa yang baik. **Logistic Regression** terbukti menjadi model terbaik secara keseluruhan dengan **akurasi 80.45% dan F1-score 59.86%**, diikuti oleh **Support Vector Machine (SVM)** dan **Random Forest**, yang juga menunjukkan performa seimbang namun sedikit lebih rendah.

Model **Bernoulli Naive Bayes** mencatat **recall tertinggi (79.86%)**, menjadikannya pilihan ideal untuk mendeteksi sebanyak mungkin pelanggan yang berisiko churn, meskipun memiliki precision yang rendah. Teknik **encoding fitur kategorikal** dan **normalisasi numerik** memberikan kontribusi besar terhadap stabilitas model, terutama pada model-model seperti SVM dan Logistic Regression.

Dibandingkan dengan studi literatur, hasil Logistic Regression dalam proyek ini menunjukkan keseimbangan yang baik antara sensitivitas (recall) dan ketepatan (precision), menjadikannya pilihan yang layak untuk implementasi bisnis nyata. Meski begitu, performa model masih dapat ditingkatkan lebih lanjut melalui pendekatan seperti:
- **Hyperparameter tuning** pada Random Forest dan SVM.
- **Oversampling** untuk menangani distribusi target `Churn` yang tidak seimbang.
- **Feature selection** untuk menyederhanakan model tanpa mengorbankan performa.

Dengan pendekatan ini, perusahaan dapat secara proaktif mempertahankan pelanggan berisiko tinggi dengan strategi personalisasi seperti diskon loyalitas atau peningkatan layanan, dan pada akhirnya **menurunkan churn rate serta meningkatkan Customer Lifetime Value (CLV)**.



