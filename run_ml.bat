@echo off
setlocal enabledelayedexpansion

set "VER_ARG=1"
set "VER_DESC=Version 1 (Raw Mean + Variance Baseline, Recommended)"
set "EXTRA_ARGS="
set "CLEAN_FLAG="

:: Check all arguments for version (1 or 2) and clean mode
for %%a in (%*) do (
    if "%%a"=="1" (
        set "VER_ARG=1"
        set "VER_DESC=Version 1 (Raw Mean + Variance Baseline, Recommended)"
    ) else if /i "%%a"=="v1" (
        set "VER_ARG=1"
        set "VER_DESC=Version 1 (Raw Mean + Variance Baseline, Recommended)"
    ) else if "%%a"=="2" (
        set "VER_ARG=2"
        set "VER_DESC=Version 2 (CLAHE Contrast Normalized)"
    ) else if /i "%%a"=="v2" (
        set "VER_ARG=2"
        set "VER_DESC=Version 2 (CLAHE Contrast Normalized)"
    ) else if /i "%%a"=="clean" (
        set "CLEAN_FLAG=--clean"
    ) else if /i "%%a"=="--clean" (
        set "CLEAN_FLAG=--clean"
    ) else if /i "%%a"=="-c" (
        set "CLEAN_FLAG=--clean"
    ) else (
        set "EXTRA_ARGS=!EXTRA_ARGS! %%a"
    )
)

echo ==========================================================
echo  Starting ML Auto-Schlieren
echo  Active Model : %VER_DESC%
if defined CLEAN_FLAG (
    echo  Display Mode : CLEAN MODE [Headless - Direct Robotic Alignment]
) else (
    echo  Display Mode : WINDOW MODE [OpenCV HUD Window Enabled]
)
echo  Executing    : python schlieren_ml/ml_auto_schlieren.py --version %VER_ARG% %CLEAN_FLAG% !EXTRA_ARGS!
echo ==========================================================

python "%~dp0schlieren_ml\ml_auto_schlieren.py" --version %VER_ARG% %CLEAN_FLAG% !EXTRA_ARGS!
