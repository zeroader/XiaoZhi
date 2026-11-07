"""Build a single EXE plus editable settings/models, then ZIP for distribution."""

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent
ENV = ROOT / ".packaging-venv"
PYTHON = ENV / "Scripts" / "python.exe"
DIST = ROOT / "dist"
PACKAGE = DIST / "XiaozhiServer"


def run(*args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Reuse installed build dependencies")
    parser.add_argument("--include-local-config", action="store_true",
                        help="Copy local lamp/server config (may contain device credentials)")
    args = parser.parse_args()
    if sys.platform != "win32" or struct.calcsize("P") != 8:
        parser.error("Build this release using 64-bit Python on Windows.")
    temp = ROOT / ".packaging-tmp"
    temp.mkdir(exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temp)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYINSTALLER_CONFIG_DIR"] = str(ROOT / ".packaging-work" / "cache")
    if not args.skip_install:
        if not PYTHON.is_file():
            venv.create(ENV, with_pip=True)
        run(PYTHON, "-m", "pip", "install", "-r", ROOT / "build_requirements.txt")
    if not PYTHON.is_file():
        parser.error("No build environment found. Run without --skip-install first.")

    models = ROOT.parent / "models"
    if not (models / "face_detection_yunet_2023mar.onnx").is_file():
        parser.error("Required face model is missing from the project models directory.")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    run(PYTHON, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", PACKAGE, "--workpath", ROOT / ".packaging-work",
        ROOT / "windows_server.spec")
    for name in ("server_config", "lamp_config"):
        local = ROOT / f"{name}.json"
        source = local if args.include_local_config and local.is_file() else ROOT / f"{name}.example.json"
        shutil.copy2(source, PACKAGE / f"{name}.json")
        shutil.copy2(ROOT / f"{name}.example.json", PACKAGE / f"{name}.example.json")
    # Models stay editable/replacable and are never extracted to a temporary path.
    shutil.copytree(models, PACKAGE / "models", dirs_exist_ok=True)
    shutil.copy2(ROOT / "PORTABLE_README.md", PACKAGE / "使用说明.md")
    (PACKAGE / "启动服务.bat").write_bytes(
        b'@echo off\r\ncd /d "%~dp0"\r\n"%~dp0XiaozhiServer.exe" %*\r\npause\r\n')
    with (PACKAGE / "build-versions.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([str(PYTHON), "-m", "pip", "freeze"], stdout=stream, check=True)
    archive = shutil.make_archive(str(DIST / "XiaozhiServer-Windows-x64"), "zip", DIST, PACKAGE.name)
    print(f"\nEXE: {PACKAGE / 'XiaozhiServer.exe'}\nZIP: {archive}")


if __name__ == "__main__":
    main()
