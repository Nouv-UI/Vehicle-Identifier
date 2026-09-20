# Program Identifikasi Jenis & Warna Kendaraan

Program Python untuk mengidentifikasi dan mendeteksi berbagai jenis kendaraan serta warna dominan dari gambar/foto maupun kamera real-time menggunakan teknologi YOLOv8 (deep learning), OpenCV, dan Matplotlib.

## 📋 Daftar Isi
- [Fitur](#fitur)
- [Persyaratan](#persyaratan)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
- [Contoh Implementasi](#contoh-implementasi)
- [API Reference](#api-reference)
- [Tips & Trik](#tips--trik)

---

## ✨ Fitur

✅ **Deteksi Jenis Kendaraan** - Mengidentifikasi mobil, motor, bus, truk, dan perahu secara presisi  
✅ **Deteksi Warna Dominan** - Mengidentifikasi warna kendaraan (Merah, Biru, Hijau, Kuning, Hitam, Putih, dll.) menggunakan analisis ruang warna HSV  
✅ **Kamera Real-time (Webcam)** - Deteksi langsung dari kamera laptop/webcam secara *real-time*  
✅ **Visualisasi Diagram Matplotlib** - Otomatis menampilkan diagram batang (*bar chart*) jenis kendaraan dan diagram lingkaran (*pie chart*) persentase warna  
✅ **Visualisasi Bounding Box** - Gambar hasil dengan kotak pembatas, label jenis, warna, dan confidence score  
✅ **Batch Processing** - Proses banyak gambar sekaligus dalam satu direktori  
✅ **Confidence Score & Koordinat Presisi** - Informasi lengkap tingkat kepercayaan dan lokasi kendaraan  

---

## 📦 Persyaratan

- Python 3.8 atau lebih baru
- pip (Python package manager)
- Webcam (opsional, untuk fitur deteksi real-time)
- ~2GB storage untuk model YOLOv8 (auto-download saat pertama kali dijalankan)

---

## 🔧 Instalasi

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python matplotlib
```

### 2. Unduh Program

Letakkan file `vehicle_identifier.py` di folder kerja Anda.

### 3. Verifikasi Instalasi

```python
from vehicle_identifier import VehicleIdentifier

# Test
identifier = VehicleIdentifier()
print("✓ Instalasi berhasil!")
```

---

## 🚀 Cara Penggunaan

### Penggunaan Interaktif (Menu Utama)

```bash
python vehicle_identifier.py
```

Program akan menampilkan menu interaktif:
```
============================================================
PROGRAM IDENTIFIKASI JENIS & WARNA KENDARAAN
============================================================

Pilihan:
1. Identifikasi dari satu gambar
2. Identifikasi dan visualisasi dari satu gambar
3. Proses seluruh direktori
4. Identifikasi real-time dari kamera laptop (webcam)
5. Keluar
```

### Penggunaan dalam Script Python

#### Contoh 1: Identifikasi Gambar Tunggal (Jenis & Warna)

```python
from vehicle_identifier import VehicleIdentifier

# Inisialisasi
identifier = VehicleIdentifier(model_name='yolov8n.pt')

# Identifikasi jenis dan warna
result = identifier.identify_vehicle('mobil.jpg')

# Tampilkan hasil
print(f"Total kendaraan: {result['total_vehicles']}")
for vehicle in result['vehicles']:
    print(f"  - {vehicle['type']} (Warna: {vehicle['color']}): {vehicle['confidence']}%")
```

**Output:**
```
Total kendaraan: 2
  - Mobil (Warna: Merah): 95.45%
  - Mobil (Warna: Hitam): 87.23%
```

#### Contoh 2: Identifikasi + Visualisasi Bounding Box & Diagram Grafik

```python
# Identifikasi, gambar hasil, dan tampilkan grafik Matplotlib
result = identifier.identify_and_visualize(
    image_path='mobil.jpg',
    output_path='hasil_deteksi.jpg'
)
```

#### Contoh 3: Identifikasi Real-time dari Kamera Laptop

```python
# Jalankan stream kamera laptop (tekan 'q' atau ESC untuk keluar)
identifier.identify_realtime(camera_index=0)
```

#### Contoh 4: Tampilkan Diagram Grafik Statistik Manual

```python
vehicles_data = [
    {'type': 'Mobil', 'color': 'Merah'},
    {'type': 'Mobil', 'color': 'Hitam'},
    {'type': 'Motor', 'color': 'Biru'}
]

# Tampilkan diagram batang & pie chart Matplotlib
identifier.show_statistics_chart(vehicles_data, title_suffix="(Sampel)")
```

---

## 📚 API Reference

### Class: VehicleIdentifier

#### Constructor
```python
VehicleIdentifier(model_name='yolov8n.pt')
```

#### Method: identify_vehicle()
```python
identify_vehicle(image_path)
```

**Return Value:**
```python
{
    'image_path': str,
    'total_vehicles': int,
    'vehicles': [
        {
            'type': str,           # Jenis kendaraan (Mobil, Motor, Bus, Truk, Perahu)
            'color': str,          # Warna dominan (Merah, Hitam, Putih, Biru, dll.)
            'confidence': float,   # 0-100
            'coordinates': {
                'x1': int, 'y1': int,
                'x2': int, 'y2': int,
                'width': int, 'height': int
            }
        }
    ]
}
```

#### Method: identify_and_visualize()
```python
identify_and_visualize(image_path, output_path=None)
```

#### Method: identify_realtime()
```python
identify_realtime(camera_index=0)
```
Membuka kamera laptop/webcam untuk deteksi real-time. Tekan `'q'` atau `'ESC'` untuk menutup kamera dan menampilkan grafik statistik Matplotlib.

#### Method: show_statistics_chart()
```python
show_statistics_chart(vehicles_data, title_suffix="")
```
Menampilkan diagram batang (*bar chart*) jenis kendaraan dan diagram lingkaran (*pie chart*) persentase warna menggunakan Matplotlib.

#### Method (Static): detect_color()
```python
VehicleIdentifier.detect_color(crop_img)
```
Mendeteksi warna dominan pada citra kendaraan menggunakan analisis ruang warna HSV.

---

## 🎯 Jenis Kendaraan yang Dideteksi

| No | Jenis Kendaraan | Nama Inggris | Deskripsi |
|----|---|---|---|
| 1 | Mobil | Car | Kendaraan roda 4 penumpang |
| 2 | Motor | Motorcycle | Kendaraan roda 2 |
| 3 | Bus | Bus | Kendaraan roda banyak penumpang |
| 4 | Truk | Truck | Kendaraan roda banyak barang |
| 5 | Perahu | Boat | Kendaraan air |

---

## 🎨 Warna Kendaraan yang Dideteksi

- 🔴 **Merah**
- 🟡 **Kuning**
- 🟢 **Hijau**
- 🔵 **Biru**
- 🟠 **Oranye**
- ⬛ **Hitam**
- ⬜ **Putih**
- 🔘 **Abu-abu / Perak**

---

## 🔧 Troubleshooting

### Error: "No module named 'matplotlib'"
```bash
pip install matplotlib
```

### Error: "No module named 'ultralytics'"
```bash
pip install ultralytics opencv-python
```

---

**Last Updated:** September 2026  
**Version:** 1.2 (Added Color Recognition, Webcam Stream, & Matplotlib Charts)

