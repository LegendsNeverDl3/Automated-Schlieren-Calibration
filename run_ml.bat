@echo off
setlocal

set "VERSION=%~1"
if "%VERSION%"=="" set "VERSION=1"

set "VER_ARG=1"
set "VER_DESC=Version 1 (Raw Mean + Variance Baseline, Recommended)"
set "HAS_VER=0"

if "%VERSION%"=="1" (
    set "VER_ARG=1"
    set "HAS_VER=1"
) else if /i "%VERSION%"=="v1" (
    set "VER_ARG=1"
    set "HAS_VER=1"
) else if "%VERSION%"=="2" (
    set "VER_ARG=2"
    set "VER_DESC=Version 2 (CLAHE Contrast Normalized)"
    set "HAS_VER=1"
) else if /i "%VERSION%"=="v2" (
    set "VER_ARG=2"
    set "VER_DESC=Version 2 (CLAHE Contrast Normalized)"
    set "HAS_VER=1"
)

set "EXTRA_ARGS="
if "%HAS_VER%"=="1" (
    for /f "tokens=1* delims= " %%a in ("%*") do set "EXTRA_ARGS=%%b"
) else (
    set "EXTRA_ARGS=%*"
)

echo ==========================================================
echo  Starting ML Auto-Schlieren
echo  Active Model: %VER_DESC%
echo  Executing   : python schlieren_ml/ml_auto_schlieren.py --version %VER_ARG% %EXTRA_ARGS%
echo ==========================================================

python "%~dp0schlieren_ml\ml_auto_schlieren.py" --version %VER_ARG% %EXTRA_ARGS%
