@echo off
chcp 65001 >nul
echo ============================================
echo   Сборка MLP Digit Recognition (.exe)
echo   Windows — PyInstaller
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ОШИБКА] Python не найден. Установите Python 3.10+ с python.org
    pause
    exit /b 1
)

echo [1/4] Установка зависимостей...
pip install tensorflow numpy Pillow pyinstaller --quiet
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось установить зависимости
    pause
    exit /b 1
)

echo [2/4] Обучение модели (если нет mnist_mlp.h5)...
if not exist "mnist_mlp.h5" (
    python -c "from model import MLPModel; m = MLPModel(); acc = m.train(); print(f'Точность: {acc*100:.2f}%%')"
)

echo [3/4] Сборка EXE с PyInstaller...
pyinstaller --onefile ^
            --windowed ^
            --name "MLP_Digits" ^
            --add-data "mnist_mlp.h5;." ^
            --hidden-import=tensorflow ^
            --hidden-import=PIL ^
            main.py

if %errorlevel% neq 0 (
    echo [ОШИБКА] Сборка не удалась
    pause
    exit /b 1
)

echo [4/4] Готово!
echo.
echo  Файл: dist\MLP_Digits.exe
echo  Размер может быть ~300-500 MB (TensorFlow включён)
echo.
pause
