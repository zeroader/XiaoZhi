$ErrorActionPreference = 'Stop'

# Restore tracked files to origin/main, then remove files introduced by the UI migration.
git apply -R .\ui-on-main.rollback.patch
Remove-Item -LiteralPath .\main\assets\vision -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\scripts\prepare_vision_dashboard.py -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\scripts\prepare_vision_runtime_assets.py -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\server\detectors\face_posture_detector.py -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\ui-on-main.rollback.patch -Force -ErrorAction SilentlyContinue
Write-Host 'UI migration rolled back to the origin/main working tree.'
