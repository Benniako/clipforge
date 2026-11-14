"""OCR back-ends for the scoreboard tracker, plus the string clean-up they need.

Three engines, each measured on real HUD pixels:

* **RapidOCR** (recognition only, PP-OCRv6 small, models bundled; used first
  when installed) — reads both the score digits and the ``MM:SS`` clock
  line. On the 16.6-min EA FC recording it gave the same 9 goals and minutes
  as EasyOCR + Tesseract, faster (13 s against 16 s); on the 40-min Valorant
  match it read 829 of 1174 score plates against EasyOCR's 596.
* **EasyOCR** (recognition only, no text detection) — read every isolated bold
  score digit correctly, where Tesseract returned nothing for 0, 4, 5 and 6.
* **Tesseract** (the plain ``tesseract`` binary; no Python wrapper needed) —
  read the ``MM:SS`` clock line correctly, where EasyOCR turned the colon into
  an "8".

Any one alone is enough: the tracker fills digits the engine cannot read
from the rule that a score only ever goes up by one, and the clock parser
repairs a colon that came back as a digit.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Protocol

log = logging.getLogger("clipforge.hud.readers")

DIGITS = "0123456789"
CLOCK_CHARS = "0123456789:+"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Reader(Protocol):
    name: str

    def read_digits(self, img) -> tuple[str, float]:
        """Text of a one- or two-digit image and a 0..1 confidence."""

    def read_line(self, img, charset: str) -> tuple[str, float]:
        """Text of a single line restricted to ``charset`` and a 0..1 confidence."""


def clean_digits(text: str) -> int | None:
    """A score value from raw OCR text, or ``None`` when it is not one.

    Accepts one or two digits. Anything else (empty, letters, three or more
    characters) is not a plausible football score digit and is discarded
    rather than guessed at.
    """
    s = (text or "").strip()
    if 1 <= len(s) <= 2 and s.isdigit():
        return int(s)
    return None


_CLOCK_RE = re.compile(r"^(\d{1,3}):(\d{2})$")


def parse_clock(text: str) -> tuple[int, int, int] | None:
    """``(minutes, seconds, added_minutes)`` from a match-clock string.

    Handles ``12:29``, stoppage time ``90:56+2``, a colon the engine dropped
    (``1229``) and a colon read as a digit (EasyOCR returns ``12829``). Returns
    ``None`` when the text is not a plausible clock.
    """
    s = re.sub(r"\s+", "", text or "")
    if not s:
        return None
    added = 0
    if "+" in s:
        main, _, extra = s.partition("+")
        extra = extra.strip("+")
        if extra.isdigit() and len(extra) <= 2:
            added = int(extra)
        s = main
    m = _CLOCK_RE.match(s)
    if m:
        mm, ss = int(m.group(1)), int(m.group(2))
    elif s.isdigit() and len(s) == 4:            # colon dropped: MMSS
        mm, ss = int(s[:2]), int(s[2:])
    elif s.isdigit() and len(s) in (5, 6):       # colon read as a digit: MM?SS / MMM?SS
        mm, ss = int(s[:-3]), int(s[-2:])
    else:
        return None
    if ss >= 60 or mm > 130:
        return None
    return mm, ss, added


def match_minute(clock: tuple[int, int, int] | None) -> int | None:
    """Football minute for a clock reading (``12:29`` is the 13th minute)."""
    if clock is None:
        return None
    return clock[0] + 1


# --------------------------------------------------------------------------- #
# EasyOCR — recognition only
# --------------------------------------------------------------------------- #
class EasyOcrReader:
    name = "easyocr"

    def __init__(self) -> None:
        import easyocr

        gpu = False
        try:
            import torch

            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
        # detector=False: only the small recognition model is loaded; the
        # tracker always tells the recogniser exactly where the text is.
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False, detector=False)

    def _recognize(self, img, allow: str) -> tuple[str, float]:
        if img is None:
            return "", 0.0
        h, w = img.shape[:2]
        res = self._reader.recognize(img, horizontal_list=[[0, w, 0, h]], free_list=[],
                                     allowlist=allow, detail=1)
        if not res:
            return "", 0.0
        return str(res[0][1]).strip(), max(0.0, min(1.0, float(res[0][2])))

    def read_digits(self, img) -> tuple[str, float]:
        return self._recognize(img, DIGITS)

    def read_line(self, img, charset: str) -> tuple[str, float]:
        return self._recognize(img, charset)


# --------------------------------------------------------------------------- #
# RapidOCR — recognition only (PP-OCRv6 small, bundled with the package)
# --------------------------------------------------------------------------- #
# Letters the recogniser returns for digit shapes; applied only to characters
# outside the wanted charset, since RapidOCR has no allow-list of its own.
_DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1",
                                   "|": "1", "S": "5", "s": "5", "B": "8"})


def restrict(text: str, charset: str) -> str:
    """``text`` limited to ``charset``, after mapping digit look-alikes."""
    mapped = "".join(ch if ch in charset else ch.translate(_DIGIT_LOOKALIKES)
                     for ch in (text or "").replace(" ", ""))
    return "".join(ch for ch in mapped if ch in charset)


class RapidOcrReader:
    """RapidOCR without its text detector: the tracker already knows where the
    text is. Measured on the EA FC clock strip: ``35:32`` at 0.9998 in 9.5 ms
    a read on the CPU (314 ms with the detector), where EasyOCR read the colon
    as an 8."""
    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr import RapidOCR

        self._engine = RapidOCR(params={"Global.log_level": "warning",
                                        "Global.use_det": False, "Global.use_cls": False})

    def _recognize(self, img, charset: str) -> tuple[str, float]:
        if img is None or getattr(img, "size", 0) == 0:
            return "", 0.0
        res = self._engine(img, use_det=False, use_cls=False, use_rec=True)
        txts = getattr(res, "txts", None) or ()
        scores = getattr(res, "scores", None) or ()
        if not txts:
            return "", 0.0
        conf = float(scores[0]) if scores else 0.0
        return restrict(str(txts[0]), charset), max(0.0, min(1.0, conf))

    def read_digits(self, img) -> tuple[str, float]:
        return self._recognize(img, DIGITS)

    def read_line(self, img, charset: str) -> tuple[str, float]:
        return self._recognize(img, charset)


# --------------------------------------------------------------------------- #
# Tesseract — the command-line binary
# --------------------------------------------------------------------------- #
def _tessdata_dir(exe: str) -> str | None:
    """A tessdata directory that really contains ``eng.traineddata``.

    ``TESSDATA_PREFIX`` is honoured first, but a prefix pointing at an empty or
    missing folder (seen in the wild) must not disable an otherwise working
    install, so the folder next to the binary and the usual system locations
    are tried as well.
    """
    cands: list[Path] = []
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        cands += [Path(env), Path(env) / "tessdata"]
    cands += [Path(exe).resolve().parent / "tessdata",
              Path("/usr/share/tesseract-ocr/5/tessdata"),
              Path("/usr/share/tesseract-ocr/4.00/tessdata"),
              Path("/usr/share/tessdata"),
              Path("/usr/local/share/tessdata"),
              Path("/opt/homebrew/share/tessdata")]
    for c in cands:
        try:
            if (c / "eng.traineddata").is_file():
                return str(c)
        except OSError:
            continue
    return None


def parse_tsv(tsv: str) -> tuple[str, float]:
    """Joined word text and the best word confidence (0..1) from Tesseract TSV."""
    text, conf = "", 0.0
    for line in tsv.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 12 and cols[0] == "5" and cols[11].strip():
            text += cols[11].strip()
            try:
                conf = max(conf, float(cols[10]) / 100.0)
            except ValueError:
                pass
    return text, max(0.0, min(1.0, conf))


class TesseractCliReader:
    name = "tesseract"

    def __init__(self, exe: str | None = None) -> None:
        if exe is None:
            from ...config import _find_executable

            exe = _find_executable("tesseract")
        if not exe:
            raise RuntimeError("tesseract binary not found")
        data = _tessdata_dir(exe)
        if not data:
            raise RuntimeError("tesseract found but eng.traineddata is missing")
        self._exe = exe
        self._data = data

    def _run(self, img, psm: int, charset: str) -> tuple[str, float]:
        import cv2

        if img is None:
            return "", 0.0
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return "", 0.0
        cmd = [self._exe, "stdin", "stdout", "--tessdata-dir", self._data, "-l", "eng",
               "--oem", "1", "--psm", str(psm),
               "-c", f"tessedit_char_whitelist={charset}", "tsv"]
        try:
            proc = subprocess.run(cmd, input=buf.tobytes(), capture_output=True,
                                  timeout=30, creationflags=_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("tesseract call failed: %s", e)
            return "", 0.0
        if proc.returncode != 0:
            return "", 0.0
        return parse_tsv(proc.stdout.decode("utf-8", "replace"))

    def read_digits(self, img) -> tuple[str, float]:
        return self._run(img, 10, DIGITS)       # psm 10: a single character

    def read_line(self, img, charset: str) -> tuple[str, float]:
        return self._run(img, 7, charset)       # psm 7: a single text line


def pick_readers() -> tuple[Reader | None, Reader | None]:
    """``(digit_reader, line_reader)`` from what is installed.

    RapidOCR reads both when installed. Otherwise digits prefer EasyOCR, the
    clock line prefers Tesseract, and whichever engine is missing is covered
    by the other. ``(None, None)`` when none works — the tracker then relies
    on the score-only-goes-up rule alone.
    """
    try:
        rapid = RapidOcrReader()
        return rapid, rapid
    except Exception as e:
        log.info("hud: RapidOCR not usable (%s)", e)
    easy: Reader | None = None
    tess: Reader | None = None
    try:
        easy = EasyOcrReader()
    except Exception as e:
        log.info("hud: EasyOCR not usable (%s)", e)
    try:
        tess = TesseractCliReader()
    except Exception as e:
        log.info("hud: Tesseract not usable (%s)", e)
    return (easy or tess), (tess or easy)
