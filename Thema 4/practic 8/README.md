# Практична робота № 8 «Класифікація». Варіант 7

`make_classification`: `n_samples = 7000`, `n_classes = 9` → поділ і масштабування →
нейронна мережа Keras із Softmax-виходом → навчання з анімацією → accuracy,
`classification_report`, матриця плутанини.

## Сумісність

TensorFlow 2.21.0 (остання стабільна версія) має wheel-файли для Windows лише для
**Python 3.10–3.13**. На Python 3.14 `pip install tensorflow` не знайде пакет, тому середовище створюється з **Python 3.13**

## Структура

```
pr8_classification/
├── main.py
├── requirements.txt
├── README.md
└── results/            # створюється автоматично під час запуску
```

## 1. Встановлення Python 3.13 (якщо його немає)

Перевірити встановлені версії:

```powershell
py --list
```

Якщо 3.13 відсутній — один із варіантів:

```powershell
py install 3.13
```

(команда нового Python Install Manager) або інсталятор Windows x86-64 з
https://www.python.org/downloads/ (гілка 3.13; залиште увімкненим «tcl/tk and IDLE» —
потрібно для вікон Matplotlib).

TensorFlow на Windows також потребує Microsoft Visual C++ Redistributable
(зазвичай уже встановлений).

## 2. Віртуальне середовище та залежності (PowerShell у терміналі VS Code)

Відкрийте папку проєкту у VS Code: **File → Open Folder…**, далі **Terminal → New Terminal**.

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`python --version` має показати `Python 3.13.x`.

Перевірка TensorFlow:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

У VS Code оберіть інтерпретатор: `Ctrl+Shift+P` → **Python: Select Interpreter** →
`.\.venv\Scripts\python.exe`.

## 3. Запуск

```powershell
python main.py
```

Під час навчання відкривається вікно з графіками accuracy і loss, яке оновлюється
після кожної епохи. Після завершення відкриються підсумкові графіки; програма
закінчує роботу, коли закрити всі вікна.

## 4. Результати (`results/`)

| Файл                      | Вміст                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| `class_distribution.png`    | розподіл зразків за 9 класами                               |
| `training_animation.gif`    | анімація accuracy/loss по епохах (PillowWriter, без FFmpeg)      |
| `training_history.png`      | підсумкові графіки accuracy і loss                                |
| `confusion_matrix.png`      | матриця плутанини на тестовій вибірці              |
| `classification_report.txt` | test accuracy, звіт класифікації, матриця плутанини |
