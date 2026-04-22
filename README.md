# MLP — Распознавание цифр (MNIST)
## Курсовой проект

### Описание
Приложение распознаёт рукописные цифры (0–9) с помощью многослойного перцептрона (MLP), обученного на датасете MNIST.

### Архитектура сети
```
Input (784)  →  Dense(256, ReLU)  →  BatchNorm  →  Dropout(0.3)
             →  Dense(128, ReLU)  →  BatchNorm  →  Dropout(0.2)
             →  Dense(64,  ReLU)  →  Dropout(0.1)
             →  Dense(10, Softmax)
```

### Быстрый старт

#### 1. Установить зависимости
```bash
pip install -r requirements.txt
```

#### 2. Запустить приложение
```bash
python main.py
```
При первом запуске нажмите **«Обучить модель»** — обучение займёт ~2-5 минут.  
Модель сохранится в файл `mnist_mlp.h5` и будет загружаться автоматически.

### Сборка исполняемого файла

#### Windows (.exe)
```
Запустите build_windows.bat от имени обычного пользователя
```

#### macOS (.app)
```bash
chmod +x build_macos.sh
./build_macos.sh
```

### Структура файлов
```
mlp_project/
├── main.py            ← Главный файл (GUI)
├── model.py           ← MLP модель (TensorFlow/Keras)
├── requirements.txt   ← Зависимости
├── build_windows.bat  ← Сборка для Windows
├── build_macos.sh     ← Сборка для macOS
└── mnist_mlp.h5       ← Обученная модель (создаётся после обучения)
```
