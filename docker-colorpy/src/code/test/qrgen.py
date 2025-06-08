""" import qrcode
from PIL import Image
import os
import time

# Zeitmessung starten
start_time = time.time()

# Zielordner für die QR-Codes
output_dir = "/Users/heikopieper/Documents/test"
os.makedirs(output_dir, exist_ok=True)

# Grundeinstellungen
base_string = "ealcfA2f-782f-4120-а316-c4c966869a83"
target_size_mm = 30
dpi = 300
mm_per_inch = 25.4
target_size_px = int((target_size_mm / mm_per_inch) * dpi)

# Anzahl der zu generierenden Codes
count = 1100

for i in range(1, count + 1):
    # Neuen Daten-String mit laufender Nummer
    data = f"{base_string}_{i:05d}"

    # QR-Code erzeugen
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    # Auf Zielgröße skalieren
    img_resized = img.resize((target_size_px, target_size_px), Image.LANCZOS)

    # Datei speichern
    filename = f"qr_code_{i:05d}.png"
    filepath = os.path.join(output_dir, filename)
    img_resized.save(filepath, dpi=(dpi, dpi))

    # Optional: Fortschritt anzeigen
    if i % 100 == 0:
        print(f"{i} QR-Codes gespeichert...")

# Zeitmessung beenden
end_time = time.time()
elapsed_time = end_time - start_time

# Ergebnis anzeigen
print(f"Fertig: {count} QR-Codes wurden generiert.")
print(f"Dauer: {elapsed_time:.2f} Sekunden")
 """
 
 
""" 
import qrcode
from PIL import Image
import os
import time
from multiprocessing import Pool, cpu_count

# === Konfiguration ===
output_dir = "/Users/heikopieper/Documents/test"
base_string = "ealcfA2f-782f-4120-а316-c4c966869a83"
target_size_mm = 30
dpi = 300
count = 11000

# === Konstante Berechnung nur 1x ===
mm_per_inch = 25.4
target_size_px = int((target_size_mm / mm_per_inch) * dpi)
os.makedirs(output_dir, exist_ok=True)

# === Worker-Funktion für parallele Ausführung ===
def generate_qr(index):
    data = f"{base_string}_{index:05d}"
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_resized = img.resize((target_size_px, target_size_px), Image.LANCZOS)
    filename = f"qr_code_{index:05d}.png"
    filepath = os.path.join(output_dir, filename)
    img_resized.save(filepath, dpi=(dpi, dpi))
    return index  # Für Fortschrittsanzeige

# === Hauptprogramm mit Zeitmessung ===
if __name__ == "__main__":
    start_time = time.time()

    # Anzahl Prozesse auf verfügbare Kerne begrenzen
    num_workers = cpu_count()  # z.B. 8 auf M2
    print(f"Starte mit {num_workers} parallelen Prozessen...")

    with Pool(processes=num_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(generate_qr, range(1, count + 1)), 1):
            if i % 100 == 0:
                print(f"{i} QR-Codes gespeichert...")

    elapsed_time = time.time() - start_time
    print(f"Fertig: {count} QR-Codes wurden generiert.")
    print(f"Dauer: {elapsed_time:.2f} Sekunden")
"""


""" import segno
from multiprocessing import Pool, cpu_count
import os
import time

# === Konfiguration ===
output_dir = "/Users/heikopieper/Documents/test"
os.makedirs(output_dir, exist_ok=True)

base_string = "ealcfA2f-782f-4120-а316-c4c966869a83"
count = 11000
target_size_mm = 30
dpi = 300

def get_scale(qr, target_mm, dpi):
    size_in_px = (target_mm / 25.4) * dpi
    modules = qr.symbol_size()[0] + 8  # call the method!
    return max(1, int(size_in_px / modules))

def generate_qr(index):
    data = f"{base_string}_{index:05d}"
    qr = segno.make(data, error='m')

    scale = get_scale(qr, target_size_mm, dpi)

    filename = f"qr_code_{index:05d}.png"
    filepath = os.path.join(output_dir, filename)

    qr.save(filepath, kind='png', scale=scale, border=4, dpi=dpi)
    return index

if __name__ == "__main__":
    start = time.time()
    with Pool(cpu_count()) as pool:
        for i, _ in enumerate(pool.imap_unordered(generate_qr, range(1, count + 1)), 1):
            if i % 100 == 0:
                print(f"{i} QR-Codes gespeichert...")
    print(f"Fertig in {time.time() - start:.2f} Sekunden") """
    
    
import segno
from multiprocessing import Pool, cpu_count
import os
import time
from functools import partial

# === Configuration ===
OUTPUT_DIR = "/Users/heikopieper/Documents/test02"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_STRING = "ealcfA2f-782f-4120-а316-c4c966869a83"
COUNT = 11000
TARGET_SIZE_MM = 30
DPI = 100
BATCH_SIZE = 100  # Process this many items before reporting progress

# Pre-calculate constants
PX_PER_MM = DPI / 25.4
TARGET_SIZE_PX = TARGET_SIZE_MM * PX_PER_MM

def get_scale(qr):
    """Calculate scale factor for QR code to reach target size."""
    modules = qr.symbol_size()[0] + 8  # QR code size including border
    return max(1, int(TARGET_SIZE_PX / modules))

def generate_qr_batch(start_end):
    """Generate a batch of QR codes to reduce file system overhead."""
    start, end = start_end
    for index in range(start, end + 1):
        data = f"{BASE_STRING}_{index:05d}"
        qr = segno.make(data, error='m')
        scale = get_scale(qr)
        
        filename = f"qr_code_{index:05d}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        qr.save(filepath, kind='png', scale=scale, border=4, dpi=DPI)
    return (start, end)

def create_batches(total, batch_size):
    """Create batches of indices to process."""
    return [(i, min(i + batch_size - 1, total)) 
            for i in range(1, total + 1, batch_size)]

if __name__ == "__main__":
    start_time = time.time()
    num_workers = cpu_count()
    
    # Create batches to reduce progress reporting overhead
    batches = create_batches(COUNT, BATCH_SIZE)
    
    with Pool(num_workers) as pool:
        for i, (batch_start, batch_end) in enumerate(pool.imap_unordered(generate_qr_batch, batches), 1):
            print(f"Completed batch {i}/{len(batches)} (QR codes {batch_start}-{batch_end})")
    
    total_time = time.time() - start_time
    qr_per_sec = COUNT / total_time
    print(f"Generated {COUNT} QR codes in {total_time:.2f} seconds ({qr_per_sec:.1f} QR codes/sec)")
