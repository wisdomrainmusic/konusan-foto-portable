Konuşan Foto – Reels Generator

Rtx Pc Planı Yapıldı. Portable bir hale getirildi.

SadTalker + FFmpeg + PyQt6 UI Pipeline

Projenin Amacı

Tek bir dikey fotoğraf ve ses (wav) dosyasından:

Omuz/kafa hareketi minimum

Sadece yüz konuşan

1080x1920 Instagram Reels uyumlu

Yüksek kaliteli (512 render)
bir video üretmek.

Tüm süreç tek tıkla çalışan bir UI üzerinden yürütülür.

Genel Mimari
konusan-foto/
│
├─ sadtalker/              # SadTalker repo (orijinal)
│   └─ inference.py
│
├─ konusan-ui/             # UI + pipeline
│   ├─ ui_app.py
│   ├─ run_pipeline.py
│   ├─ config.py
│   └─ run.bat             # (sonradan eklenecek)
│
├─ input/
│   ├─ photo.jpg
│   └─ audio.wav
│
├─ output_ui/
│   └─ YYYY_MM_DD_xx.xx.xx/
│       ├─ temp_*.mp4
│       └─ reels.mp4
│
└─ ffmpeg/
    └─ ffmpeg.exe

Kullanılan Teknolojiler

Python 3.10

SadTalker (face animation)

FFmpeg (static build)

PyQt6 (desktop UI)

PowerShell / Windows

Ana Akış (Pipeline)
1️⃣ SadTalker Çalıştırma

Fotoğraf + ses alınır, SadTalker ile konuşan yüz üretilir.

Önemli ayarlar:

--still → kafa/omuz hareketi minimum

--preprocess full → dikey foto kadrajını korur

--size 512 → render kalitesi yükseltilir

--enhancer none → stabilite için kapalı

SadTalker bazen exit code 1 döndürse bile video ürettiği için:

check=False kullanıldı

Çıktı olarak temp_*.mp4 manuel aranır

2️⃣ Reels (1080x1920) Üretimi – FFmpeg

SadTalker çıktısı Instagram Reels formatına çevrilir:

Crop YOK

Aspect ratio korunur

Gerekirse pad ile 1080x1920 tamamlanır

Lanczos scale (daha net görüntü)

Encode ayarları:

libx264

preset slow

crf 16 (yüksek kalite)

yuv420p

+faststart

run_pipeline.py Özeti

İki ana fonksiyon vardır:

run_sadtalker()

SadTalker’ı subprocess ile çalıştırır

temp_*.mp4 dosyasını otomatik bulur

Hata verse bile çıktı varsa devam eder

make_reels()

FFmpeg ile reels çıktısını üretir

1080x1920 sabit

Yüksek kalite encode

UI (PyQt6)
UI Özellikleri

📷 Foto seç

🎧 Ses seç (wav)

🚀 Render Reels (1080x1920)

Log alanı (ileride eklenecek)

UI, run_pipeline.py içindeki fonksiyonları çağırır.

Karşılaşılan Problemler ve Çözümler
❌ PowerShell parametre hataları

Sebep: Komutları satır satır çalıştırmak
Çözüm: Tüm parametreler Python listesi içinde verildi

❌ Wrong model version none

Sebep: --enhancer none SadTalker’da exit 1 tetikliyor
Çözüm:

check=False

Video gerçekten oluşmuş mu diye manuel kontrol

❌ Kafa dışında her şey kırpılıyor

Sebep: Varsayılan SadTalker crop
Çözüm:

--preprocess full

Crop işlemi tamamen FFmpeg’e bırakıldı

Mevcut Durum (Checkpoint)

✅ Foto + ses → konuşan yüz
✅ Omuz/kafa sabit
✅ Dikey kadraj korunuyor
✅ Reels çıktısı doğru
✅ UI çalışıyor
✅ Kalite 512 render + yüksek bitrate

Sonraki Adımlar (Yeni Sohbette)
UI’de şu anda kalan eksikler

PowerShell log/progress görünürlüğü: run.bat ile açınca ilerleme akmıyor gibi (log yönlendirme/flush meselesi).

Kalite: 512 çıktı hedefi (--size 512) + gerekirse GFPGAN/RealESRGAN stratejisi.

“Tutorial’daki web UI kalitesi” benzeri ayarların (crop/resize/full still/face enhancer) UI’da seçenekli hale getirilmesi.

7) Pipeline dosyası (run_pipeline.py) – güncel yaklaşım

SadTalker çağrısı check=False (bazı durumlarda hata verip video üretiyor).

Çıktı arama: output_dir/**/temp_*.mp4 newest seç.

Reels: crop yerine pad ile 1080×1920 (dikey foto kadrajını bozmamak için).

8) Kalite hedefi (sonraki adım)

SadTalker komutuna --size 512 ekleyip gerçek çözünürlüğü artıracağız.

Ayrıca:

Reels encode’da -crf değerini düşürmek (ör. 16/14) kaliteyi artırır ama dosya büyür.

Kaynağın dikey foto olduğu varsayımını koruyarak pad’li pipeline en stabil yöntem.

9) RTX’li PC planı

Aynı yapı RTX’li bilgisayarda denenerek:

Torch CUDA kurulumu

daha hızlı render / potansiyel kalite artışı hedeflenecek.

Yeni sohbet için “Devam Planı”

run_pipeline.py içine --size 512 ekle (repo destekliyorsa) + UI’den toggle.

UI log akışını düzelt (subprocess stdout/stderr’i UI text box’a gerçek zamanlı bas).

run.bat ile tek tık çalıştırma:

venv aktivasyon + pip check + UI start

hata olursa pencerede kalıp log göstersin.

Opsiyonel: GFPGAN aç/kapat seçeneği (ve “Wrong model version none” durumuna dayanıklı try/catch).

RTX PC’ye kurulum checklist’i (CUDA torch + aynı checkpoint yapısı).

 UI’ya Kalite seçimi (256 / 512) ekleme

 run.bat ile tek tık çalıştırma

 Log ekranı (progress + hata)

 EXE build (PyInstaller)

 Preset profilleri (Reels / Shorts / TikTok)

Not

Bu proje üretim seviyesine çok yakın bir prototip haline gelmiştir.

Yeni sohbette bu README referans alınarak direkt geliştirmeye devam edilebilir.
