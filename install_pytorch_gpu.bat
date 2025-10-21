@echo off
echo ============================================================
echo PyTorch GPU Kurulum Scripti
echo ============================================================
echo.

echo [1/3] Mevcut PyTorch'u kaldiriyor...
pip uninstall torch torchvision torchaudio -y

echo.
echo [2/3] PyTorch GPU versiyonunu kuruyor (CUDA 11.8)...
echo Bu islem 5-10 dakika surebilir...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/3] Test ediliyor...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
echo Kurulum tamamlandi!
echo ============================================================
pause


