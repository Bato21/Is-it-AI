"""
deck_a_imagenes.py - Convierte presentaciones PDF / PPTX en una imagen PNG
por diapositiva.

Uso
---
    # convierte presentaciones_fuente/ -> sin_clasificar/ (ambas bajo herramientas/)
    python herramientas/deck_a_imagenes.py

    # convierte un archivo puntual
    python herramientas/deck_a_imagenes.py "C:/ruta/a/presentacion.pptx"

    # mayor resolución y carpeta de salida distinta
    python herramientas/deck_a_imagenes.py --dpi 200 --out sin_clasificar

Cómo funciona
-------------
PDF  -> se renderiza directo con PyMuPDF (fitz). No necesita software externo.
PPTX -> primero se convierte a PDF y luego se renderiza. Para el paso PPTX->PDF
        se usa, en orden:
          1. Microsoft PowerPoint vía COM   (Windows + PowerPoint instalado)
          2. LibreOffice `soffice --headless` (cualquier SO, si está instalado)
        Si no hay ninguno, exporta la presentación a PDF a mano y vuelve a correr.

Las imágenes se nombran  <nombre_deck>__slide_03.png  para poder rastrear cada
diapositiva hasta su presentación de origen mientras las clasificas.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # consola UTF-8 en Windows
except Exception:
    pass

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Falta dependencia: PyMuPDF. Corre  pip install -r herramientas/requirements.txt")

AQUI = Path(__file__).resolve().parent
ENTRADA_DEF = AQUI / "presentaciones_fuente"
SALIDA_DEF = AQUI / "sin_clasificar"


# --------------------------------------------------------------------------- #
# PPTX -> PDF
# --------------------------------------------------------------------------- #
def pptx_a_pdf_powerpoint(src: Path, out_dir: Path) -> Path | None:
    """Convierte con Microsoft PowerPoint vía COM. Devuelve ruta PDF o None."""
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return None
    pdf_path = out_dir / (src.stem + ".pdf")
    powerpoint = None
    try:
        powerpoint = win32com.client.Dispatch("PowerPoint.Application")
        deck = powerpoint.Presentations.Open(str(src), WithWindow=False)
        deck.SaveAs(str(pdf_path), 32)  # 32 = ppSaveAsPDF
        deck.Close()
        return pdf_path if pdf_path.exists() else None
    except Exception as exc:  # noqa: BLE001
        print(f"  PowerPoint COM falló: {exc}")
        return None
    finally:
        if powerpoint is not None:
            try:
                powerpoint.Quit()
            except Exception:  # noqa: BLE001
                pass


def pptx_a_pdf_libreoffice(src: Path, out_dir: Path) -> Path | None:
    """Convierte con LibreOffice headless. Devuelve ruta PDF o None."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(src)],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"  LibreOffice falló: {exc}")
        return None
    pdf_path = out_dir / (src.stem + ".pdf")
    return pdf_path if pdf_path.exists() else None


def pptx_a_pdf(src: Path, out_dir: Path) -> Path | None:
    return pptx_a_pdf_powerpoint(src, out_dir) or pptx_a_pdf_libreoffice(src, out_dir)


# --------------------------------------------------------------------------- #
# PDF -> PNG
# --------------------------------------------------------------------------- #
def pdf_a_pngs(pdf: Path, out_dir: Path, stem: str, dpi: int) -> int:
    doc = fitz.open(pdf)
    zoom = dpi / 72.0  # 72 = DPI base del PDF
    matrix = fitz.Matrix(zoom, zoom)
    n = 0
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)
        pix.save(out_dir / f"{stem}__slide_{i:02d}.png")
        n += 1
    doc.close()
    return n


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def convertir(src: Path, out_dir: Path, dpi: int) -> int:
    ext = src.suffix.lower()
    print(f"-> {src.name}")
    if ext == ".pdf":
        return pdf_a_pngs(src, out_dir, src.stem, dpi)
    if ext in {".pptx", ".ppt"}:
        with tempfile.TemporaryDirectory() as tmp:
            pdf = pptx_a_pdf(src, Path(tmp))
            if pdf is None:
                print(
                    "  No se pudo convertir el PPTX. Instala Microsoft PowerPoint o\n"
                    "  LibreOffice, o exporta la presentación a PDF a mano y déjala\n"
                    "  en herramientas/presentaciones_fuente/."
                )
                return 0
            return pdf_a_pngs(pdf, out_dir, src.stem, dpi)
    print(f"  Tipo de archivo no soportado: {ext}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Convierte PDF/PPTX en PNG por diapositiva.")
    ap.add_argument(
        "entradas",
        nargs="*",
        help="Archivos. Por defecto: todo lo de herramientas/presentaciones_fuente/",
    )
    ap.add_argument("--out", default=str(SALIDA_DEF), help="Carpeta de salida de los PNG.")
    ap.add_argument("--dpi", type=int, default=150, help="Resolución de render (def 150).")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.entradas:
        decks = [Path(p) for p in args.entradas]
    else:
        decks = (
            sorted(
                p for p in ENTRADA_DEF.iterdir() if p.suffix.lower() in {".pdf", ".pptx", ".ppt"}
            )
            if ENTRADA_DEF.exists()
            else []
        )

    if not decks:
        print(f"No hay presentaciones. Deja archivos .pdf / .pptx en {ENTRADA_DEF}")
        return

    total = 0
    for deck in decks:
        if not deck.exists():
            print(f"!! No existe: {deck}")
            continue
        total += convertir(deck, out_dir, args.dpi)

    print(f"\nListo. {total} imagen(es) de diapositiva escritas en {out_dir}")
    print("Siguiente: clasifícalas en dataset/{0_sin_ia, 1_rastro_ia, 2_saturada_ia}/")


if __name__ == "__main__":
    main()
