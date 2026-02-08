# run_pipeline.py
import os
import glob
import subprocess
from pathlib import Path
from config import *


def _newest_temp_mp4(search_root: str) -> str | None:
    mp4s = glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)
    mp4s = [m for m in mp4s if os.path.basename(m).lower().startswith("temp_")]
    if not mp4s:
        return None
    mp4s.sort(key=lambda p: os.path.getmtime(p))
    return mp4s[-1]


def run_sadtalker(image_path: str, audio_path: str, output_dir: str) -> str:
    """
    SadTalker'ı çalıştırır.
    Not: Bazı ortamlarda enhancer='none' yüzünden SadTalker en sonda exit 1 dönebiliyor,
    ama video üretmiş oluyor. Bu yüzden check=True kullanmıyoruz; çıktı var mı diye bakıyoruz.
    """
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        SADTALKER_PYTHON,
        "inference.py",
        "--driven_audio", str(Path(audio_path).resolve()),
        "--source_image", str(Path(image_path).resolve()),
        "--result_dir", output_dir,

        # ✅ omuz/kafa hareketini minimum
        "--still",
        "--preprocess", "full",

        # ✅ kalite
        "--size", "512",

        # ✅ enhancer kapalı
        "--enhancer", "none",
    ]

    p = subprocess.run(
        cmd,
        cwd=SADTALKER_DIR,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    found = _newest_temp_mp4(output_dir)
    if found:
        return found

    alt = _newest_temp_mp4(os.path.join(SADTALKER_DIR, output_dir))
    if alt:
        return alt

    tail = (p.stdout or "")[-2000:]
    raise RuntimeError(
        f"SadTalker çıktı üretmedi. returncode={p.returncode}\n\n--- log tail ---\n{tail}"
    )


def make_reels(video_path: str, audio_path: str, reels_path: str) -> str:
    """
    Klasik reels (scale+pad). Zoom sorunu çözmez.
    """
    reels_path = str(Path(reels_path).resolve())
    os.makedirs(os.path.dirname(reels_path), exist_ok=True)

    vf = (
        "scale=1080:1920:flags=lanczos:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    )

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(Path(video_path).resolve()),
        "-i", str(Path(audio_path).resolve()),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        reels_path
    ]

    subprocess.run(cmd, check=True)
    return reels_path


def make_reels_fullbody_overlay(face_video_path: str, image_path: str, audio_path: str, reels_path: str) -> str:
    """
    ✅ Full Body Background + Talking Face Overlay (MASK'li)

    - Arka plan: Orijinal foto (1080x1920 scale+pad) -> kadraj korunur
    - Üst katman: SadTalker video (yüz konuşur)
    - Kenarlar: FEATHER MASK ile yumuşatılır (dikdörtgen hissi azalır)
    """
    import cv2
    import numpy as np

    reels_path = str(Path(reels_path).resolve())
    os.makedirs(os.path.dirname(reels_path), exist_ok=True)

    img_path = str(Path(image_path).resolve())
    face_video_path = str(Path(face_video_path).resolve())
    audio_path = str(Path(audio_path).resolve())

    # 1) Foto oku
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("Foto okunamadı. Yol veya dosya bozuk olabilir.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Yüz tespit
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        raise RuntimeError("Yüz tespit edilemedi. Daha net/önden bir foto dene.")

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    # 3) 1080x1920 scale+pad sonrası yüz bbox konumu
    H, W = img.shape[:2]
    target_w, target_h = 1080, 1920

    scale = min(target_w / W, target_h / H)
    newW, newH = int(W * scale), int(H * scale)
    padX = (target_w - newW) // 2
    padY = (target_h - newH) // 2

    X = int(x * scale) + padX
    Y = int(y * scale) + padY
    WW = int(w * scale)
    HH = int(h * scale)

    # 4) Overlay alanı (normal)
    expand = 0.35
    ex = int(WW * expand)
    ey = int(HH * expand)

    ox = max(0, X - ex)
    oy = max(0, Y - ey)
    ow = min(target_w - ox, WW + 2 * ex)
    oh = min(target_h - oy, HH + 2 * ey)

    # 5) FEATHER MASK üret (mask.png)
    mask_path = str(Path(reels_path).with_name("mask.png"))

    feather = int(min(ow, oh) * 0.10)  # %10 yumuşatma
    feather = max(10, feather)

    yy, xx = np.mgrid[0:oh, 0:ow]
    dist_left = xx
    dist_right = (ow - 1) - xx
    dist_top = yy
    dist_bottom = (oh - 1) - yy
    dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))

    alpha = np.clip((dist / feather) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(mask_path, alpha)

    # 6) ffmpeg overlay (alphamerge)
    bg_vf = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2"

    cmd = [
        FFMPEG_PATH, "-y",

        # bg (foto)
        "-loop", "1",
        "-i", img_path,

        # fg (SadTalker video)
        "-i", face_video_path,

        # audio
        "-i", audio_path,

        # mask (alpha)
        "-loop", "1",
        "-i", mask_path,

        "-filter_complex",
        f"[0:v]{bg_vf}[bg];"
        f"[1:v]scale={ow}:{oh},eq=gamma=0.95:contrast=0.95,format=rgba[fg];"
        f"[3:v]scale={ow}:{oh},format=gray[msk];"
        f"[fg][msk]alphamerge[fgm];"
        f"[bg][fgm]overlay={ox}:{oy}:format=auto[v]",

        "-map", "[v]",
        "-map", "2:a",

        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",

        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        reels_path
    ]

    subprocess.run(cmd, check=True)
    return reels_path
