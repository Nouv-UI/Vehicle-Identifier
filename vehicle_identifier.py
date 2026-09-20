"""
Program Identifikasi Jenis Kendaraan dari Gambar
Menggunakan YOLOv8 untuk deteksi dan klasifikasi kendaraan
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from collections import Counter

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Silahkan install ultralytics terlebih dahulu")
    print("Jalankan: pip install ultralytics opencv-python matplotlib")
    sys.exit(1)


class VehicleIdentifier:
    """Kelas untuk identifikasi jenis kendaraan dari gambar"""
    
    # Mapping ID kelas kendaraan dari COCO dataset
    VEHICLE_CLASSES = {
        2: "Mobil",
        3: "Motor",
        5: "Bus",
        7: "Truk",
        9: "Perahu",
    }
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        Inisialisasi model YOLO
        
        Args:
            model_name: Nama model YOLO (nano, small, medium, large, xlarge)
        """
        print(f"Loading model YOLO: {model_name}...")
        try:
            self.model = YOLO(model_name)
            print("✓ Model berhasil dimuat")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
        
        self.confidence_threshold = 0.5
    
    @staticmethod
    def show_statistics_chart(vehicles_data, title_suffix=""):
        """
        Menampilkan diagram batang jenis kendaraan & pie chart warna menggunakan Matplotlib
        
        Args:
            vehicles_data: List of dicts berisi info kendaraan [{'type': ..., 'color': ...}, ...]
            title_suffix: String tambahan untuk judul grafik
        """
        if plt is None:
            print("\n✗ Library 'matplotlib' belum terpasang. Jalankan: pip install matplotlib")
            return
            
        if not vehicles_data:
            print("\nℹ Tidak ada kendaraan terdeteksi untuk dibuatkan grafik statistik.")
            return
        
        all_types = [v['type'] for v in vehicles_data if 'type' in v]
        all_colors = [v['color'] for v in vehicles_data if 'color' in v]
        
        if not all_types:
            print("\nℹ Tidak ada data kendaraan terdeteksi.")
            return

        type_counts = Counter(all_types)
        color_counts = Counter(all_colors)

        # Buat figure dengan 2 subplot (1 baris, 2 kolom)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        window_title = f"Statistik Deteksi Kendaraan {title_suffix}".strip()
        fig.canvas.manager.set_window_title(window_title)
        fig.suptitle(window_title, fontsize=14, fontweight='bold')

        # Subplot 1: Bar Chart (Jenis Kendaraan)
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        bars = ax1.bar(types, counts, color='#4C72B0', edgecolor='black')
        ax1.set_title('Jumlah per Jenis Kendaraan', fontsize=12)
        ax1.set_xlabel('Jenis Kendaraan')
        ax1.set_ylabel('Jumlah Kendaraan')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.yaxis.get_major_locator().set_params(integer=True)
        
        # Angka di atas bar
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        # Subplot 2: Pie Chart (Warna Kendaraan)
        colors_labels = list(color_counts.keys())
        colors_values = list(color_counts.values())
        
        color_map = {
            "Merah": "#E15759", "Kuning": "#EDC948", "Hijau": "#59A14F",
            "Biru": "#4E79A7", "Oranye": "#F28E2B", "Hitam": "#404040",
            "Putih": "#E0E0E0", "Abu-abu": "#BAB0AC", "Tidak diketahui": "#76B7B2"
        }
        pie_colors = [color_map.get(c, "#A0A0A0") for c in colors_labels]
        
        ax2.pie(colors_values, labels=colors_labels, colors=pie_colors, autopct='%1.1f%%', startangle=140)
        ax2.set_title('Persentase Warna Kendaraan', fontsize=12)

        print("\n>>> Menampilkan grafik statistik Matplotlib...")
        print(">>> Petunjuk: Tutup jendela grafik untuk kembali ke menu utama.\n")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def detect_color(crop_img):
        """
        Deteksi warna dominan pada area crop kendaraan menggunakan ruang warna HSV
        
        Args:
            crop_img: Image patch (Numpy array BGR) dari area kendaraan
            
        Returns:
            String nama warna (Merah, Kuning, Hijau, Biru, Oranye, Hitam, Putih, Abu-abu/Perak, atau Tidak diketahui)
        """
        if crop_img is None or crop_img.size == 0:
            return "Tidak diketahui"
        
        # Ambil area tengah saja (50% tengah) untuk meminimalkan pengaruh ban, kaca, dan latar belakang
        h, w, _ = crop_img.shape
        start_h, end_h = int(h * 0.25), int(h * 0.75)
        start_w, end_w = int(w * 0.25), int(w * 0.75)
        
        if end_h > start_h and end_w > start_w:
            center_crop = crop_img[start_h:end_h, start_w:end_w]
        else:
            center_crop = crop_img

        hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        
        # Range warna HSV
        color_ranges = {
            "Merah": [
                (np.array([0, 70, 50]), np.array([10, 255, 255])),
                (np.array([170, 70, 50]), np.array([180, 255, 255]))
            ],
            "Kuning": [(np.array([15, 70, 50]), np.array([35, 255, 255]))],
            "Hijau": [(np.array([36, 50, 50]), np.array([85, 255, 255]))],
            "Biru": [(np.array([90, 50, 50]), np.array([130, 255, 255]))],
            "Oranye": [(np.array([11, 70, 50]), np.array([24, 255, 255]))],
            "Hitam": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            "Putih": [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
            "Abu-abu": [(np.array([0, 0, 50]), np.array([180, 50, 199]))],
        }
        
        max_pixels = 0
        detected_color = "Tidak diketahui"
        
        for color_name, ranges in color_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges:
                mask |= cv2.inRange(hsv, lower, upper)
            
            pixel_count = cv2.countNonZero(mask)
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                detected_color = color_name
                
        return detected_color

    def identify_vehicle(self, image_path):
        """
        Identifikasi jenis dan warna kendaraan dari gambar
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Dictionary berisi hasil identifikasi
        """
        # Validasi file gambar
        if not Path(image_path).exists():
            print(f"✗ File tidak ditemukan: {image_path}")
            return None
        
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Tidak bisa membaca gambar: {image_path}")
            return None
        
        # Jalankan deteksi
        results = self.model(image, conf=self.confidence_threshold)
        
        # Proses hasil deteksi
        vehicles_found = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter hanya kendaraan
                if class_id in self.VEHICLE_CLASSES:
                    vehicle_type = self.VEHICLE_CLASSES[class_id]
                    
                    # Ambil koordinat bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop area kendaraan untuk analisis warna
                    crop_img = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                    vehicle_color = self.detect_color(crop_img)
                    
                    vehicles_found.append({
                        'type': vehicle_type,
                        'color': vehicle_color,
                        'confidence': round(confidence * 100, 2),
                        'coordinates': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    })
        
        return {
            'image_path': image_path,
            'total_vehicles': len(vehicles_found),
            'vehicles': vehicles_found
        }
    
    def identify_and_visualize(self, image_path, output_path=None):
        """
        Identifikasi kendaraan (jenis & warna) dan gambar hasil dengan bounding box
        
        Args:
            image_path: Path ke file gambar input
            output_path: Path untuk menyimpan gambar hasil (opsional)
        """
        # Baca gambar
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Tidak bisa membaca gambar: {image_path}")
            return None
        
        # Jalankan deteksi
        results = self.model(image, conf=self.confidence_threshold)
        
        # Salin gambar untuk visualisasi custom
        annotated_image = image.copy()
        
        vehicles_found = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id in self.VEHICLE_CLASSES:
                    vehicle_type = self.VEHICLE_CLASSES[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop area kendaraan untuk deteksi warna
                    crop_img = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
                    vehicle_color = self.detect_color(crop_img)
                    
                    # Gambar custom bounding box dan label (Jenis - Warna)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{vehicle_type} ({vehicle_color}) - {confidence*100:.1f}%"
                    cv2.putText(annotated_image, label, (x1, max(y1-10, 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    vehicles_found.append({
                        'type': vehicle_type,
                        'color': vehicle_color,
                        'confidence': round(confidence * 100, 2)
                    })
        
        # Tampilkan gambar
        cv2.imshow('Vehicle Identification & Color Detection', annotated_image)
        print("\nTekan tombol apapun untuk menutup gambar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Simpan gambar hasil (jika output_path diberikan)
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"✓ Gambar hasil disimpan ke: {output_path}")
        
        return {
            'total_vehicles': len(vehicles_found),
            'vehicles': vehicles_found
        }
    
    def process_directory(self, directory_path):
        """
        Proses semua gambar dalam direktori
        
        Args:
            directory_path: Path ke direktori berisi gambar
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = []
        
        # Cari semua file gambar
        for ext in image_extensions:
            image_files.extend(Path(directory_path).glob(f'*{ext}'))
            image_files.extend(Path(directory_path).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"✗ Tidak ada gambar ditemukan di: {directory_path}")
            return []
        
        print(f"\nMemproses {len(image_files)} gambar...\n")
        results = []
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Memproses: {image_path.name}")
            result = self.identify_vehicle(str(image_path))
            if result:
                results.append(result)
                self._print_result(result)
        
        return results
    
    def identify_realtime(self, camera_index=0):
        """
        Identifikasi jenis dan warna kendaraan secara real-time menggunakan kamera/webcam
        
        Args:
            camera_index: Indeks kamera (default: 0 untuk kamera bawaan laptop)
        """
        print(f"\nMembuka kamera (index: {camera_index})...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"✗ Gagal membuka kamera dengan indeks {camera_index}.")
            return
        
        print("✓ Kamera berhasil dibuka!")
        print(">>> Petunjuk: Tekan tombol 'q' atau 'ESC' pada jendela kamera untuk keluar.\n")
        
        session_vehicles = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Gagal mengambil frame dari kamera.")
                break
            
            # Jalankan deteksi
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            vehicle_count = 0
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in self.VEHICLE_CLASSES:
                        vehicle_type = self.VEHICLE_CLASSES[class_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicle_count += 1
                        
                        # Crop area kendaraan untuk deteksi warna real-time
                        crop_img = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        vehicle_color = self.detect_color(crop_img)
                        
                        session_vehicles.append({
                            'type': vehicle_type,
                            'color': vehicle_color,
                            'confidence': round(confidence * 100, 2)
                        })
                        
                        # Gambar bounding box dan label (Jenis - Warna)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{vehicle_type} ({vehicle_color}) - {confidence*100:.1f}%"
                        cv2.putText(frame, label, (x1, max(y1-10, 20)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tampilkan informasi jumlah kendaraan di pojok atas frame
            info_text = f"Kendaraan Terdeteksi: {vehicle_count} | Tekan 'q' untuk keluar"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Tampilkan frame di jendela OpenCV
            cv2.imshow('Real-time Vehicle & Color Identification', frame)
            
            # Cek tombol tekan ('q' atau ESC (27))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Menutup kamera...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Tampilkan grafik statistik dari sesi kamera real-time setelah kamera ditutup
        if session_vehicles:
            self.show_statistics_chart(session_vehicles, title_suffix="(Sesi Kamera Real-time)")
    
    @staticmethod
    def _print_result(result):
        """Cetak hasil identifikasi dengan format rapi"""
        print(f"  Total kendaraan ditemukan: {result['total_vehicles']}")
        
        if result['vehicles']:
            for idx, vehicle in enumerate(result['vehicles'], 1):
                color_info = f", Warna: {vehicle['color']}" if 'color' in vehicle else ""
                print(f"    {idx}. {vehicle['type']}{color_info} "
                      f"(Confidence: {vehicle['confidence']}%)")
        print()


def main():
    """Fungsi utama dengan contoh penggunaan"""
    
    # Inisialisasi identifier
    identifier = VehicleIdentifier(model_name='yolov8n.pt')
    
    print("\n" + "="*60)
    print("PROGRAM IDENTIFIKASI JENIS & WARNA KENDARAAN")
    print("="*60)
    
    while True:
        print("\nPilihan:")
        print("1. Identifikasi dari satu gambar")
        print("2. Identifikasi dan visualisasi dari satu gambar")
        print("3. Proses seluruh direktori")
        print("4. Identifikasi real-time dari kamera laptop (webcam)")
        print("5. Keluar")
        
        choice = input("\nMasukkan pilihan (1-5): ").strip()
        
        if choice == '1':
            image_path = input("Masukkan path gambar: ").strip()
            result = identifier.identify_vehicle(image_path)
            
            if result:
                print("\n" + "="*60)
                print("HASIL IDENTIFIKASI")
                print("="*60)
                identifier._print_result(result)
                if result['vehicles']:
                    identifier.show_statistics_chart(result['vehicles'], title_suffix="(Gambar Tunggal)")
        
        elif choice == '2':
            image_path = input("Masukkan path gambar: ").strip()
            output_path = input("Path penyimpanan hasil (tekan Enter untuk skip): ").strip()
            
            result = identifier.identify_and_visualize(
                image_path,
                output_path if output_path else None
            )
            
            if result:
                print("\n" + "="*60)
                print("HASIL IDENTIFIKASI")
                print("="*60)
                print(f"Total kendaraan ditemukan: {result['total_vehicles']}")
                for idx, vehicle in enumerate(result['vehicles'], 1):
                    color_info = f", Warna: {vehicle['color']}" if 'color' in vehicle else ""
                    print(f"  {idx}. {vehicle['type']}{color_info} "
                          f"(Confidence: {vehicle['confidence']}%)")
                if result['vehicles']:
                    identifier.show_statistics_chart(result['vehicles'], title_suffix="(Gambar Tunggal)")
        
        elif choice == '3':
            directory_path = input("Masukkan path direktori: ").strip()
            results = identifier.process_directory(directory_path)
            
            if results:
                print("\n" + "="*60)
                print("RINGKASAN KESELURUHAN")
                print("="*60)
                total_vehicles = sum(r['total_vehicles'] for r in results)
                print(f"Total gambar diproses: {len(results)}")
                print(f"Total kendaraan ditemukan: {total_vehicles}")
                
                # Kumpulkan semua kendaraan dari seluruh gambar di direktori
                all_dir_vehicles = []
                for res in results:
                    all_dir_vehicles.extend(res.get('vehicles', []))
                
                if all_dir_vehicles:
                    identifier.show_statistics_chart(all_dir_vehicles, title_suffix="(Direktori)")
        
        elif choice == '4':
            identifier.identify_realtime()
        
        elif choice == '5':
            print("\nTerima kasih! Program selesai.")
            break
        
        else:
            print("✗ Pilihan tidak valid. Silahkan coba lagi.")


if __name__ == "__main__":
    main()