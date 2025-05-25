# Laporan Proyek Machine Learning Terapan - Reksi Hendra Pratama

## Domain Proyek: Telekomunikasi

### Latar Belakang
Industri telekomunikasi merupakan salah satu industri yang menghasilkan data dalam jumlah sangat besar setiap harinya. Salah satu permasalahan utama dalam industri ini adalah **customer churn** — kondisi ketika pelanggan memutuskan untuk berhenti berlangganan layanan. Kehilangan pelanggan bukan hanya menurunkan pendapatan tetapi juga meningkatkan biaya akuisisi pelanggan baru.

Menurut Wagh et al. (2023), biaya mendapatkan pelanggan baru bisa 5 kali lebih besar daripada mempertahankan pelanggan lama, sehingga fokus perusahaan seharusnya diarahkan pada **strategi retensi pelanggan berbasis prediksi**. Dengan pendekatan machine learning, perusahaan dapat memanfaatkan data histori pelanggan untuk mengidentifikasi siapa yang kemungkinan akan churn dan mengambil tindakan preventif secara tepat waktu.

Model prediksi churn telah menunjukkan hasil yang menjanjikan. Studi oleh Wagh et al. menggunakan Random Forest dan menghasilkan akurasi hingga **99.09%**, menandakan kekuatan model ini untuk mendukung keputusan bisnis secara strategis. Pendekatan seperti ini juga meningkatkan nilai umur pelanggan (Customer Lifetime Value) dan menjaga loyalitas mereka melalui layanan yang dipersonalisasi.

### Referensi
- Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). *Customer churn prediction in telecom using machine learning in big data platform*. Journal of Big Data, 6(28). https://doi.org/10.1186/s40537-019-0191-6
- Wagh, S. K., et al. (2023). *Customer churn prediction in telecom sector using machine learning techniques*. Results in Control and Optimization, 14, 100342. https://doi.org/10.1016/j.rico.2023.100342

---

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



