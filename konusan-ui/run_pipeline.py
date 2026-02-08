# run_pipeline.py
import os
import glob
import subprocess
import time
from pathlib import Path
from config import *
import math
import re


def _python_has_torch(python_exe: str) -> bool:
    try:
        r = subprocess.run(
            [python_exe, "-c", "import torch;print(torch.__version__)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _resolve_sadtalker_python() -> str:
    """
    Non-breaking:
    - Önce config.py içindeki SADTALKER_PYTHON denenir.
    - Torch yoksa otomatik fallback: konusan-ui/.venv/Scripts/python.exe
    """
    primary = SADTALKER_PYTHON
    if primary and Path(primary).exists() and _python_has_torch(primary):
        return primary

    # 1) SadTalker'ın kendi venv'i (en doğru)
    fallback1 = Path(SADTALKER_DIR) / ".venv" / "Scripts" / "python.exe"
    if fallback1.exists() and _python_has_torch(str(fallback1)):
        return str(fallback1)

    # 2) UI venv (varsa)
    fallback2 = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if fallback2.exists() and _python_has_torch(str(fallback2)):
        return str(fallback2)

    raise RuntimeError(
        "SadTalker python ortamında 'torch' bulunamadı.\n"
        f"- SADTALKER_PYTHON: {primary}\n"
        f"- Fallback1 (sadtalker): {fallback1}\n"
        f"- Fallback2 (ui): {fallback2}\n\n"
        "Çözüm: torch içeren python kullanın (venv) veya portable python içine torch kurun."
    )


# -------------------------
# Audio helpers (non-breaking)
# -------------------------

def _ffprobe_path() -> str:
    # ffprobe is usually shipped next to ffmpeg
    ffmpeg = Path(FFMPEG_PATH)
    cand = ffmpeg.with_name("ffprobe.exe") if ffmpeg.suffix.lower() == ".exe" else ffmpeg.with_name("ffprobe")
    return str(cand)

def _probe_duration_seconds(media_path: str) -> float | None:
    """Return duration in seconds (float). None if unknown."""
    media_path = str(Path(media_path).resolve())
    ffprobe = _ffprobe_path()
    try:
        if Path(ffprobe).exists():
            p = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", media_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                check=False
            )
            s = (p.stdout or "").strip()
            return float(s) if s else None
    except Exception:
        pass

    # fallback: parse ffmpeg -i output
    try:
        p = subprocess.run(
            [FFMPEG_PATH, "-i", media_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            check=False
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", p.stdout or "")
        if not m:
            return None
        hh, mm, ss = m.groups()
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return None

def _normalize_audio_to_wav16k(audio_path: str, work_dir: str) -> str:
    """SadTalker için sesi garanti formatta hazırlar: mono, 16kHz, PCM wav."""
    work_dir = str(Path(work_dir).resolve())
    os.makedirs(work_dir, exist_ok=True)
    out_wav = str(Path(work_dir) / "driven_audio_16k_mono.wav")

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(Path(audio_path).resolve()),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_wav
    ]
    subprocess.run(cmd, check=True)
    return out_wav

def _split_wav_into_chunks(wav_path: str, chunk_seconds: int, work_dir: str) -> list[str]:
    """wav'ı ffmpeg ile parçalara böler (ss/t)."""
    dur = _probe_duration_seconds(wav_path) or 0.0
    if dur <= 0:
        return [wav_path]

    n = int(math.ceil(dur / float(chunk_seconds)))
    chunks = []
    for i in range(n):
        ss = i * chunk_seconds
        t = min(chunk_seconds, max(0.0, dur - ss))
        out = str(Path(work_dir) / f"chunk_{i:03d}.wav")
        cmd = [
            FFMPEG_PATH, "-y",
            "-ss", str(ss),
            "-t", str(t),
            "-i", str(Path(wav_path).resolve()),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            out
        ]
        subprocess.run(cmd, check=True)
        chunks.append(out)
    return chunks

def _concat_videos(video_paths: list[str], out_path: str) -> str:
    """Video parçalarını tek videoda birleştirir (audio yok)."""
    out_path = str(Path(out_path).resolve())
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # concat filter (re-encode) -> en güvenlisi
    inputs = []
    for vp in video_paths:
        inputs += ["-i", str(Path(vp).resolve())]

    filter_complex = f"concat=n={len(video_paths)}:v=1:a=0,format=yuv420p[v]"
    cmd = [
        FFMPEG_PATH, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        out_path
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _newest_temp_mp4(search_root: str) -> str | None:
    mp4s = glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)
    mp4s = [m for m in mp4s if os.path.basename(m).lower().startswith("temp_")]
    if not mp4s:
        return None
    mp4s.sort(key=lambda p: os.path.getmtime(p))
    return mp4s[-1]

def _newest_output_mp4_after(search_root: str, t0: float) -> str | None:
    """SadTalker'ın ürettiği en yeni mp4'ü bul.

    Not: Bazı SadTalker sürümleri çıktıyı `temp_*.mp4` yerine `<timestamp>.mp4`
    olarak yazar. Bu yüzden sadece temp_ aramak false-negative üretebilir.
    """
    mp4s = glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)
    mp4s = [m for m in mp4s if os.path.getmtime(m) >= t0]

    # bizim ürettiğimiz yan çıktıları filtrele
    skip_names = {"reels.mp4", "temp_no_audio.mp4", "temp_merged.mp4"}
    mp4s = [m for m in mp4s if os.path.basename(m).lower() not in skip_names]

    # crash sonrası 0-byte dosyaları ele
    mp4s = [m for m in mp4s if os.path.getsize(m) > 0]

    if not mp4s:
        return None
    mp4s.sort(key=lambda p: os.path.getmtime(p))
    return mp4s[-1]

def _parse_audio2exp_total_frames(log_text: str) -> int | None:
    """SadTalker stdout içinden audio2exp toplam frame sayısını yakalar.
    Örn: `audio2exp: 100%|##########| 92/92 [..]`
    """
    matches = re.findall(r"audio2exp:\s*\d+%.*?\|\s*(\d+)/(\d+)\s*\[", log_text or "")
    if not matches:
        matches = re.findall(r"audio2exp:.*?(\d+)/(\d+)", log_text or "")
    if not matches:
        return None
    done, total = matches[-1]
    try:
        return int(total)
    except Exception:
        return None

def _should_force_chunking(audio_seconds: float | None, log_text: str) -> bool:
    """6 saniye dudak gibi truncation durumlarını yakalamak için heuristik."""
    if not audio_seconds:
        return False
    total_frames = _parse_audio2exp_total_frames(log_text)
    if not total_frames:
        return False
    approx_seconds = total_frames / 25.0  # SadTalker çoğunlukla ~25fps
    return (audio_seconds > 10.0) and (approx_seconds < (audio_seconds * 0.6))


def _cleanup_temp_mp4(search_root: str) -> None:
    """Stale temp mp4 dosyalarını temizler ki eski çıktı yeni sanılmasın."""
    mp4s = glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)
    mp4s = [m for m in mp4s if os.path.basename(m).lower().startswith("temp_")]
    for p in mp4s:
        try:
            os.remove(p)
        except Exception:
            pass


def _newest_temp_mp4_after(search_root: str, t0: float) -> str | None:
    """t0'dan sonra üretilen en yeni temp_*.mp4"""
    mp4s = glob.glob(os.path.join(search_root, "**", "*.mp4"), recursive=True)
    mp4s = [m for m in mp4s if os.path.basename(m).lower().startswith("temp_")]
    mp4s = [m for m in mp4s if os.path.getmtime(m) >= t0]
    if not mp4s:
        return None
    mp4s.sort(key=lambda p: os.path.getmtime(p))
    return mp4s[-1]


def run_sadtalker(image_path: str, audio_path: str, output_dir: str) -> str:
    """
    SadTalker'ı çalıştırır.
    Not: Bazı ortamlarda enhancer='none' yüzünden SadTalker en sonda exit 1 dönebiliyor,
    ama video üretmiş oluyor. Bu yüzden check=True kullanmıyoruz; çıktı var mı diye bakıyoruz.

    ✅ Non-breaking iyileştirme:
    - MP3/WAV fark etmez: sesi 16kHz mono PCM WAV'a çevirip SadTalker'a öyle verir.
    - Eğer SadTalker çıktısı sesin sadece ilk ~6 saniyesini kapsıyorsa (tipik truncation),
      otomatik olarak sesi parçalara bölüp (chunk) her parçayı ayrı renderlar ve videoları birleştirir.
    """
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    # ✅ kritik: eski temp videoları sil (stale output bug fix)
    _cleanup_temp_mp4(output_dir)

    # ✅ doğru python seç (torch kontrol)
    sadtalker_py = _resolve_sadtalker_python()

    t0 = time.time()

    work_dir = str(Path(output_dir) / "_tmp_audio")
    os.makedirs(work_dir, exist_ok=True)

    # 1) Audio'yu garanti formata getir
    norm_audio = _normalize_audio_to_wav16k(audio_path, work_dir)

    # 1.b) Checkpoint klasörü hızlı doğrula (portable paketlerde en sık hata)
    ckpt_dir = Path(SADTALKER_DIR) / "checkpoints"
    if not ckpt_dir.exists():
        raise RuntimeError(
            "SadTalker checkpoints klasörü bulunamadı:\n"
            f"- Beklenen: {ckpt_dir}\n\n"
            "Çözüm: SadTalker model dosyalarını (checkpoints/...) bu klasöre kopyalayın."
        )

    # 2) Render helper (tek deneme)
    def _run_once(
        driven_audio: str,
        out_dir: str,
        size: int,
        old_version: bool,
    ) -> tuple[str | None, str]:
        cmd = [
            sadtalker_py,
            "inference.py",
            "--driven_audio", str(Path(driven_audio).resolve()),
            "--source_image", str(Path(image_path).resolve()),
            "--result_dir", out_dir,

            # ✅ omuz/kafa hareketini minimum
            "--still",
            "--preprocess", "full",

            # ✅ kalite
            "--size", str(size),

            # ✅ enhancer kapalı
            "--enhancer", "none",
        ]

        if old_version:
            cmd.append("--old_version")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(SADTALKER_DIR).resolve()) + os.pathsep + env.get("PYTHONPATH", "")

        p = subprocess.run(
            cmd,
            cwd=SADTALKER_DIR,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Debug için log'u diske yaz (UI crash olsa bile kalsın)
        try:
            logs_dir = Path(out_dir) / "_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            (logs_dir / f"sadtalker_{stamp}_size{size}{'_old' if old_version else ''}.log").write_text(
                p.stdout or "", encoding="utf-8", errors="replace"
            )
        except Exception:
            pass

        # SadTalker çıktısı: temp_*.mp4 VEYA <timestamp>.mp4 olabilir
        found = _newest_output_mp4_after(out_dir, t0)
        if not found:
            found = _newest_output_mp4_after(os.path.join(SADTALKER_DIR, out_dir), t0)
        return found, (p.stdout or "")

    def _looks_like_checkpoint_mismatch(log_text: str) -> bool:
        needles = ["skipped mismatched keys", "adapted input_3dmm", "load_state_dict", "size mismatch"]
        t = (log_text or "").lower()
        return any(n in t for n in needles)

    # 2.a) Akıllı retry: checkpoint/versiyon uyumsuzluğu için
    a_dur = _probe_duration_seconds(norm_audio)
    attempts = [(512, False), (512, True), (256, False), (256, True)]
    found = None
    log_text = ""
    used_size = 512
    used_old_version = False

    for size, old_version in attempts:
        found, log_text = _run_once(norm_audio, output_dir, size=size, old_version=old_version)
        if found:
            used_size = size
            used_old_version = old_version
            break
        if not _looks_like_checkpoint_mismatch(log_text):
            break

    if not found:
        tail = (log_text or "")[-2200:]
        raise RuntimeError(
            "SadTalker çıktı üretmedi.\n"
            "\n--- log tail ---\n"
            f"{tail}\n\n"
            "İpucu: Bu hata genellikle yanlış/eksik checkpoint (./sadtalker/checkpoints) veya\n"
            "(size=256/512) model uyumsuzluğundan gelir. Checkpoints klasörünü doğrulayın."
        )

    # 3) Truncation kontrolü
    v_dur = _probe_duration_seconds(found)

    # Eğer süreleri ölçemediysek, mevcut davranışı bozmayalım: direkt döndür.
    if not a_dur or not v_dur:
        return found

    # Tipik bug: ~6s video, uzun audio. Burada otomatik chunk mode'a geç.
    too_short_video = (a_dur and v_dur and a_dur > 7.5 and v_dur < (a_dur - 1.0))
    looks_like_6s_lip = _should_force_chunking(a_dur, log_text)

    if too_short_video or looks_like_6s_lip:
        # 1 dakikaya kadar hedef için: 15s chunk iyi denge (4 parça)
        chunks = _split_wav_into_chunks(norm_audio, chunk_seconds=15, work_dir=work_dir)
        chunk_videos: list[str] = []

        for i, chunk in enumerate(chunks):
            chunk_out_dir = str(Path(output_dir) / f"_chunk_{i:03d}")
            os.makedirs(chunk_out_dir, exist_ok=True)
            v, chunk_log = _run_once(chunk, chunk_out_dir, size=used_size, old_version=used_old_version)
            if not v:
                tail = chunk_log[-2000:]
                raise RuntimeError(
                    f"SadTalker chunk çıktı üretmedi (chunk {i}).\n\n--- log tail ---\n{tail}"
                )
            # concat kolaylığı için: her chunk videodan audio'yu çıkar
            v_no_audio = str(Path(chunk_out_dir) / "temp_no_audio.mp4")
            subprocess.run(
                [FFMPEG_PATH, "-y", "-i", v, "-an", "-c:v", "copy", v_no_audio],
                check=True
            )
            chunk_videos.append(v_no_audio)

        merged = str(Path(output_dir) / "temp_merged.mp4")
        return _concat_videos(chunk_videos, merged)

    return found


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
