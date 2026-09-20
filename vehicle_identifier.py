"""
Program Identifikasi Jenis Kendaraan dari Gambar
Menggunakan YOLOv8 untuk deteksi dan klasifikasi kendaraan
"""

import cv2
import numpy as np
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Silahkan install ultralytics terlebih dahulu")
    print("Jalankan: pip install ultralytics opencv-python")
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
    
    def identify_vehicle(self, image_path):
        """
        Identifikasi jenis kendaraan dari gambar
        
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
                    
                    vehicles_found.append({
                        'type': vehicle_type,
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
        Identifikasi kendaraan dan gambar hasil dengan bounding box
        
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
        
        # Gambar hasil deteksi
        annotated_image = results[0].plot()
        
        vehicles_found = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id in self.VEHICLE_CLASSES:
                    vehicle_type = self.VEHICLE_CLASSES[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Gambar custom bounding box dan label
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{vehicle_type} ({confidence*100:.1f}%)"
                    cv2.putText(annotated_image, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    vehicles_found.append({
                        'type': vehicle_type,
                        'confidence': round(confidence * 100, 2)
                    })
        
        # Tampilkan gambar
        cv2.imshow('Vehicle Detection', annotated_image)
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
        Identifikasi jenis kendaraan secara real-time menggunakan kamera/webcam
        
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
                        
                        # Gambar bounding box dan label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{vehicle_type} ({confidence*100:.1f}%)"
                        cv2.putText(frame, label, (x1, max(y1-10, 20)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tampilkan informasi jumlah kendaraan di pojok atas frame
            info_text = f"Kendaraan Terdeteksi: {vehicle_count} | Tekan 'q' untuk keluar"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Tampilkan frame di jendela OpenCV
            cv2.imshow('Real-time Vehicle Identification', frame)
            
            # Cek tombol tekan ('q' atau ESC (27))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Menutup kamera...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    @staticmethod
    def _print_result(result):
        """Cetak hasil identifikasi dengan format rapi"""
        print(f"  Total kendaraan ditemukan: {result['total_vehicles']}")
        
        if result['vehicles']:
            for idx, vehicle in enumerate(result['vehicles'], 1):
                print(f"    {idx}. {vehicle['type']} "
                      f"(Confidence: {vehicle['confidence']}%)")
        print()


def main():
    """Fungsi utama dengan contoh penggunaan"""
    
    # Inisialisasi identifier
    identifier = VehicleIdentifier(model_name='yolov8n.pt')
    
    print("\n" + "="*60)
    print("PROGRAM IDENTIFIKASI JENIS KENDARAAN")
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
                    print(f"  {idx}. {vehicle['type']} "
                          f"(Confidence: {vehicle['confidence']}%)")
        
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
        
        elif choice == '4':
            identifier.identify_realtime()
        
        elif choice == '5':
            print("\nTerima kasih! Program selesai.")
            break
        
        else:
            print("✗ Pilihan tidak valid. Silahkan coba lagi.")


if __name__ == "__main__":
    main()

