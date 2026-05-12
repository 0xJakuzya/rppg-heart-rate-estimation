# rPPG Detection

Прототип для бесконтактного восстановления пульсовой волны по видео лица. Проект обрабатывает RGB-видео, выделяет несколько областей интереса на лице, собирает короткие временные окна патчей и обучает нейросетевую модель `PhysNet` восстанавливать rPPG/BVP-сигнал. Частота сердечных сокращений рассчитывается из восстановленного сигнала спектральным методом.

Проект ориентирован на исследовательские эксперименты с датасетом MCD-rPPG и демонстрацию real-time инференса с обычной веб-камеры.

## Что реализовано

- детекция лица и landmarks через MediaPipe Face Landmarker;
- выделение 8 ROI-патчей: 4 зоны лба, 2 зоны левой щеки, 2 зоны правой щеки;
- предобработка видео и синхронизированного PPG в `.npz` окна;
- нормализация патчей и PPG-сигнала;
- обучение `PhysNet` на multi-ROI патчах;
- `ShiftLoss` для учета возможной рассинхронизации видео и PPG;
- оценка качества по окнам и по восстановленному сигналу всего видео;
- webcam demo с отображением landmarks, ROI-патчей, BVP-графика и HR.

## Архитектура

```text
video + sync ppg
      |
      v
src/preprocessing.py
      |
      v
.npz windows: patches [time, roi, h, w, 3] + ppg [time]
      |
      v
src/train.py -> models/physnet.py -> cnn.pth
      |
      +--> src/evaluation.py -> predictions.csv, video_predictions.csv, summary.json
      |
      +--> src/test.py -> real-time webcam demo
```

Основной вход модели:

```text
[batch, time, roi, channels, height, width]
```

При текущих настройках:

```text
time = 300 frames
fps = 15
window = 20 seconds
roi = 8
patch = 24x24
```

## Структура проекта

```text
models/
  physnet.py        # основная нейросетевая модель
  loss.py           # ShiftLoss
  pos.py            # классический POS baseline
  chrom.py          # классический CHROM baseline

src/
  config.py         # основные параметры проекта
  face_detector.py  # landmarks, маски и ROI-патчи
  dataset.py        # загрузка .npz окон и split по пациентам
  preprocessing.py  # подготовка окон из MCD-rPPG
  train.py          # обучение PhysNet
  evaluation.py     # оценка модели
  test.py           # real-time webcam demo
  utils.py          # фильтрация, нормализация, HR estimation
  visualization.py  # отрисовка интерфейса demo
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Для GPU-обучения установите сборку PyTorch с CUDA под вашу версию драйвера. Файл `face_landmarker.task` должен лежать в корне проекта, путь задается в `src/config.py`.

## Подготовка данных

Скрипт ожидает структуру MCD-rPPG с файлом `db.csv`. В нем используются поля:

- `camera`
- `step`
- `video`
- `ppg_sync`

Пример запуска:

```bash
python -m src.preprocessing ^
  --dataset-root D:/mcd_rppg ^
  --output-dir data/mcd_rppg_windows ^
  --camera FullHDwebcam ^
  --step all ^
  --window 300 ^
  --stride 150 ^
  --frame-step 2 ^
  --video-fps 30
```

Что делает preprocessing:

- читает видео и синхронизированный PPG;
- детектирует лицо на каждом кадре;
- извлекает 8 ROI-патчей;
- заполняет пропуски по кадрам интерполяцией;
- фильтрует и нормализует PPG;
- сохраняет окна в `.npz`;
- пишет отчет `preprocessing_report.csv`.

Формат одного окна:

```text
patches: [time, roi, h, w, 3]
ppg:     [time]
```

## Обучение

Основной сценарий:

```bash
python -m src.train ^
  --data-dir data/mcd_rppg_windows ^
  --output results ^
  --epochs 15 ^
  --batch-size 4 ^
  --lr 0.0003 ^
  --num-workers 4 ^
  --model physnet ^
  --loss shiftloss ^
  --use-frame-diff ^
  --early-stopping-patience 3 ^
  --early-stopping-min-delta 0.02
```

Результаты сохраняются в новую папку `results/run_YYYYMMDD_HHMMSS`:

- `cnn.pth` - веса лучшей модели;
- `history.json` - история обучения;
- `summary.json` - параметры запуска и лучшая MAE;
- `training_curves.png` - графики обучения;
- `best_predictions.png` - примеры восстановленного BVP;
- `hr_scatter.png` - scatter и Bland-Altman.

Доступные функции потерь:

- `negpearson` - отрицательная корреляция Пирсона;
- `shiftloss` - максимум корреляции Пирсона при небольшом временном сдвиге.

Параметры `ShiftLoss` вынесены в `src/config.py`:

```python
SHIFT_LOSS_MAX_SHIFT_SEC = 0.33
SHIFT_LOSS_FPS = 15.0
SHIFT_LOSS_EPS = 1e-8
```

## Оценка

Оценка обученной модели на `.npz` окнах:

```bash
python -m src.evaluation ^
  --data-dir data/valid_41 ^
  --model-path results/best/cnn.pth ^
  --output-dir results/eval_valid_41 ^
  --model physnet ^
  --batch-size 4 ^
  --num-workers 0 ^
  --use-frame-diff
```

Выходные файлы:

- `predictions.csv` - метрики по каждому окну;
- `video_predictions.csv` - метрики после восстановления сигнала всего видео;
- `summary.json` - MAE, RMSE, bias, доля ошибок выше 5 BPM, метрики по пациентам.

Оценка поддерживает фильтрацию пациентов:

```bash
python -m src.evaluation ^
  --data-dir data/valid_41 ^
  --exclude-data-dir data/train ^
  --max-patients 20 ^
  --start-after-patient 9000 ^
  --model-path results/best/cnn.pth ^
  --output-dir results/eval_subset ^
  --use-frame-diff
```

## Real-time demo

Запуск инференса с веб-камеры:

```bash
python -m src.test ^
  --model-path results/best/cnn.pth ^
  --device cuda ^
  --use-frame-diff
```

В окне отображаются:

- landmarks лица;
- статус детекции;
- накопление окна `current/300`;
- текущая оценка HR;
- график BVP;
- отдельное окно с 8 ROI-патчами.

Выход из demo: клавиша `q`.

## Текущие экспериментальные ориентиры

По сохраненным локальным отчетам проекта:

- `PhysNet + ShiftLoss + frame difference`;
- subject-level train/validation split;
- окно 300 кадров при 15 FPS;
- лучший сохраненный held-out validation результат: `MAE 4.19 BPM`, `RMSE 6.45 BPM` на 641 окне;
- один из сохраненных video-level evaluation запусков на 41 пациенте: `video MAE 2.29 BPM`, `video RMSE 4.11 BPM`.

Эти значения являются экспериментальными и зависят от конкретного split, качества видео, набора пациентов и настроек предобработки.

## Основные настройки

Ключевые параметры находятся в `src/config.py`:

```python
FPS_TARGET = 15
CNN_WINDOW = 300
MULTI_ROI_COUNT = 8
ROI_PATCH_SIZE = 24
HR_LO_HZ = 0.75
HR_HI_HZ = 2.5
CHEBY_LO = 0.7
CHEBY_HI = 2.5
```

## Ограничения

- Это исследовательский прототип, не медицинское изделие.
- Качество зависит от освещения, движения головы, качества камеры и стабильности FPS.
- Модель требует накопления полного временного окна перед выводом устойчивого BVP/HR.
- Для корректной оценки важно разделять train/validation/test по пациентам, а не по отдельным окнам.

## Минимальный полный сценарий

```bash
# 1. подготовить окна
python -m src.preprocessing --dataset-root D:/mcd_rppg --output-dir data/mcd_rppg_windows --frame-step 2

# 2. обучить модель
python -m src.train --data-dir data/mcd_rppg_windows --loss shiftloss --use-frame-diff

# 3. оценить сохраненную модель
python -m src.evaluation --data-dir data/mcd_rppg_windows --model-path results/run_YYYYMMDD_HHMMSS/cnn.pth --output-dir results/eval --use-frame-diff

# 4. запустить webcam demo
python -m src.test --model-path results/run_YYYYMMDD_HHMMSS/cnn.pth --use-frame-diff
```

## Назначение

Проект может использоваться как база для экспериментов по:

- бесконтактной оценке ЧСС;
- восстановлению rPPG/BVP по RGB-видео;
- сравнению нейросетевых и классических методов POS/CHROM;
- анализу устойчивости rPPG к движению, освещению и выбору ROI.
