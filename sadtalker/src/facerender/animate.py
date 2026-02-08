"""SadTalker facerender animation module.

NOTE: This repository snapshot does not include the original SadTalker
implementation. The class below is a minimal placeholder to preserve the
expected import path and to apply the enhancer import guard requested.
"""

_FACE_ENHANCER_IMPORT_ERROR = None
try:
    # NOTE: enhancer=none iken bu import'a ihtiyaç yok.
    # Bazı portable kurulumlarda gfpgan/basicsr zinciri import sırasında patlayabiliyor.
    from src.utils.face_enhancer import enhancer_generator_with_len, enhancer_list
except Exception as e:
    enhancer_generator_with_len = None
    enhancer_list = None
    _FACE_ENHANCER_IMPORT_ERROR = e


class AnimateFromCoeff:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def generate(
        self,
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer="none",
        background_enhancer=None,
        preprocess="crop",
        img_size=256,
    ):
        if enhancer != "none":
            if enhancer_list is None:
                raise RuntimeError(
                    "Enhancer aktif ama face_enhancer import edilemedi. "
                    "Bu kurulumda gfpgan/basicsr bağımlılıkları eksik veya bozuk.\n\n"
                    f"Orijinal hata: {_FACE_ENHANCER_IMPORT_ERROR}"
                )
        raise NotImplementedError(
            "SadTalker src tree is not present in this repository snapshot. "
            "Please provide the original implementation of animate.py."
        )
