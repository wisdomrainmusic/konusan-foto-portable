# ui_app.py
import sys
import os
from pathlib import Path

# ✅ Portable import fix: konusan-ui + proje root'u sys.path'e ekle
THIS_DIR = Path(__file__).resolve().parent                # ...\konusan-ui
ROOT_DIR = THIS_DIR.parent                                # ...\konusan-foto_portable
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox,
    QListWidget, QListWidgetItem, QGroupBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon

# ✅ artık run_pipeline kesin bulunur
from run_pipeline import run_sadtalker, make_reels_fullbody_overlay

# ✅ config: önce root'tan dene, olmazsa konusan-ui içindekini dene
try:
    from config import DEFAULT_OUTPUT_DIR
except Exception:
    # fallback: konusan-ui/config.py varsa
    from config import DEFAULT_OUTPUT_DIR  # aynı isimle kalsın


class KonusanUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Konuşan Foto – Reels Generator")

        # Galeri + kontrol paneli için genişlik arttı
        self.setFixedSize(860, 460)

        # Drag & drop hızlandırıcı UX
        self.setAcceptDrops(True)

        self.image_path = None
        self.audio_path = None

        self.init_ui()
        self.load_gallery()

    def init_ui(self):
        root = QHBoxLayout()

        # -------------------------
        # LEFT: Preset Gallery
        # -------------------------
        left_box = QGroupBox("🖼️ Preset Gallery (presets/ klasöründen)")
        left_layout = QVBoxLayout()

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(96, 96))
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setMovement(QListWidget.Movement.Static)
        self.gallery.itemClicked.connect(self.on_gallery_click)

        left_layout.addWidget(self.gallery)
        left_box.setLayout(left_layout)

        # -------------------------
        # RIGHT: Main Controls
        # -------------------------
        right_layout = QVBoxLayout()

        self.info = QLabel("Foto ve ses seç, sonra Render'a bas.\n(İstersen sürükle-bırak da yapabilirsin.)")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.info)

        self.btn_image = QPushButton("📷 Foto Seç (jpg / jpeg / png)")
        self.btn_image.clicked.connect(self.select_image)
        right_layout.addWidget(self.btn_image)

        self.btn_audio = QPushButton("🎤 Ses Seç (mp3 / wav)")
        self.btn_audio.clicked.connect(self.select_audio)
        right_layout.addWidget(self.btn_audio)

        self.btn_render = QPushButton("🚀 Render Reels (Full Body Overlay)")
        self.btn_render.clicked.connect(self.render)
        right_layout.addWidget(self.btn_render)

        self.btn_open_out = QPushButton("📂 Output klasörünü aç")
        self.btn_open_out.clicked.connect(self.open_output_dir)
        right_layout.addWidget(self.btn_open_out)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        right_layout.addWidget(self.log)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        root.addWidget(left_box, 1)
        root.addWidget(right_widget, 2)
        self.setLayout(root)

    # -------------------------
    # Preset Gallery
    # -------------------------
    def load_gallery(self):
        """
        ROOT/presets klasörüne koyduğun jpg/jpeg/png görselleri listeler.
        Portable sistemi bozmaz: klasör yoksa oluşturur ve boş bırakır.
        """
        presets_dir = ROOT_DIR / "presets"
        presets_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png"}
        files = [p for p in presets_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: p.name.lower())

        self.gallery.clear()

        if not files:
            self.log.append(f"[INFO] Preset yok.\nŞuraya görsel at:\n{presets_dir}\n")
            return

        for p in files:
            item = QListWidgetItem(QIcon(str(p)), p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            self.gallery.addItem(item)

        self.log.append(f"[OK] {len(files)} preset yüklendi: {presets_dir}\n")

    def on_gallery_click(self, item: QListWidgetItem):
        p = item.data(Qt.ItemDataRole.UserRole)
        if p:
            self.image_path = p
            self.log.append(f"Preset seçildi:\n{p}\n")

    # -------------------------
    # File Pickers
    # -------------------------
    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Foto Seç", "", "Images (*.jpg *.jpeg *.png)"
        )
        if path:
            self.image_path = path
            self.log.append(f"Foto seçildi:\n{path}\n")

    def select_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ses Seç", "", "Audio (*.mp3 *.wav)"
        )
        if path:
            self.audio_path = path
            self.log.append(f"Ses seçildi:\n{path}\n")

    def open_output_dir(self):
        try:
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            os.startfile(DEFAULT_OUTPUT_DIR)
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Output açılamadı:\n{e}")

    # -------------------------
    # Drag & Drop UX
    # -------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        path = urls[0].toLocalFile()
        if not path:
            return

        ext = Path(path).suffix.lower()

        if ext in (".jpg", ".jpeg", ".png"):
            self.image_path = path
            self.log.append(f"Foto (drop) seçildi:\n{path}\n")
        elif ext in (".mp3", ".wav"):
            self.audio_path = path
            self.log.append(f"Ses (drop) seçildi:\n{path}\n")
        else:
            self.log.append(f"[WARN] Desteklenmeyen dosya:\n{path}\n")

    # -------------------------
    # Render (mevcut akışa dokunmuyoruz)
    # -------------------------
    def render(self):
        if not self.image_path or not self.audio_path:
            QMessageBox.warning(self, "Eksik", "Foto ve ses seçmelisin.")
            return

        try:
            self.log.append("SadTalker çalışıyor...\n")

            temp_video = run_sadtalker(
                self.image_path,
                self.audio_path,
                DEFAULT_OUTPUT_DIR
            )

            # DEBUG: SadTalker çıktısı
            self.log.append(f"[DEBUG] temp_video:\n{temp_video}\n")

            reels_path = os.path.join(DEFAULT_OUTPUT_DIR, "reels.mp4")
            self.log.append("Full body overlay reels oluşturuluyor...\n")

            make_reels_fullbody_overlay(
                temp_video,
                self.image_path,
                self.audio_path,
                reels_path
            )

            self.log.append(f"✅ Bitti!\n{reels_path}\n")
            QMessageBox.information(self, "Tamamlandı", "Reels hazır!")

            os.startfile(reels_path)

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))
            self.log.append(f"❌ Hata:\n{e}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = KonusanUI()
    ui.show()
    sys.exit(app.exec())
