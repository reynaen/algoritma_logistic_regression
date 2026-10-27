# Prediksi Pembatalan Reservasi Hotel (Logistic Regression)

TA-03 Pembelajaran Mesin, Universitas Bale Bandung.
Aplikasi web Flask untuk memprediksi apakah reservasi hotel berisiko dibatalkan.

- Dataset: Hotel Booking Demand (Kaggle) https://www.kaggle.com/jessemostipak/hotel-booking-demand
- Model: Logistic Regression + SMOTE | Accuracy 0,777 | Precision 0,695 | Recall 0,711 | F1 0,703 | AUC 0,847

## Struktur
- `TA03_Prediksi_Pembatalan_Hotel.ipynb` notebook analisis (EDA, preprocessing, model, evaluasi, penyimpanan model)
- `app.py`, `templates/index.html` aplikasi Flask
- `model/` preprocessor.joblib (scaler + encoder), logreg_model.joblib, meta.json
- `hotels.csv` dataset (isi sama dengan hotel_bookings.csv)

## Menjalankan lokal
```
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:5000

## Deploy (Render)
1. Push repositori ke GitHub.
2. Render > New > Web Service > pilih repositori.
3. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:app`.
