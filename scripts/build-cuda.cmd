@echo off
rem Builds the engine for an NVIDIA GPU on Windows: Ninja inside a Visual Studio x64 environment (the
rem Visual Studio generator mangles nvcc's arguments). Architecture 75 is Turing (RTX 20xx); pass
rem another as the first argument, e.g. 86 for an RTX 30xx.
setlocal
set ARCH=%1
if "%ARCH%"=="" set ARCH=75
rem Visual Studio 2022 first (the compiler CUDA 13 is tested with), else the newest.
set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`%VSWHERE% -version [17.0^,18.0^) -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
if not defined VS for /f "usebackq delims=" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VS=%%i
if not defined VS (
  echo Visual Studio with the C++ tools was not found
  exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cd /d "%~dp0.."
cmake -S engine -B engine/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOURIER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%ARCH% -DGGML_CUDA_FA_QUANTS=f16-f16 || exit /b 1
cmake --build engine/build-cuda || exit /b 1
