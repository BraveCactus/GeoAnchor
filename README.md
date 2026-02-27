## **Структура проекта**

### GeoAnchor/
#### │
#### ├── config.py                              # Конфигурационный файл
#### │
#### ├── inria_aerial_dataset/                  # Исходный датасет
#### │   └── AerialImageDataset/
#### │       ├── train/
#### │       │   ├── images/                    # Оригинальные TIFF (5000x5000)
#### │       │   │   ├── austin1.tif
#### │       │   │   ├── austin2.tif
#### │       │   │   ├── chicago1.tif
#### │       │   │   └── ...
#### │       │   └── gt/                        # Маски (ground truth)
#### │       └── test/
#### │           ├── images/
#### │           └── gt/
#### │
#### ├── src/                                    # Исходный код
#### │   ├── __init__.py
#### │   │
#### │   ├── data/                               # Данные города
#### │   │   └── city_data.json                  # Координаты городов
#### │   │
#### │   ├── models/                             # Модели нейросетей
#### │   │   ├── __init__.py
#### │   │   └── models.py                        # DINOv2 экстрактор
#### │   │
#### │   └── scripts/                            # Скрипты обработки
#### │       ├── __init__.py
#### │       ├── images_tools.py                  # Работа с GeoTIFF
#### │       ├── preprocess_images.py             # Предобработка
#### │       │
#### │       ├── seasons_dataset/                  # Сезонные датасеты
#### │       │   ├── __init__.py
#### │       │   ├── sn_train_data.py              # Создание сезонных данных
#### │       │   ├── sn_test_data.py               # Тестовые сезонные данные
#### │       │   ├── season_csv.py                  # CSV с эмбеддингами
#### │       │   └── seasons_map.py                 # Склейка сезонных карт
#### │       │
#### │       ├── standart_dataset/                  # Стандартные датасеты
#### │       │   ├── __init__.py
#### │       │   ├── create_dataset.py              # Создание основного датасета
#### │       │   └── create_val_data.py             # Валидационные данные
#### │       │
#### │       └── extra_datasets/                    # Дополнительные датасеты
#### │           ├── __init__.py
#### │           ├── create_chicago_dataset.py      # Датасет Чикаго
#### │           └── create_rotate_photoes.py       # Патчи с поворотом
#### │
#### ├── results/                                # Результаты работы
#### │   ├── maps/                               # Сшитые карты городов
#### │   ├── sliced_images/                       # Нарезанные патчи
#### │   ├── sliced_city_data/                     # CSV с координатами патчей
#### │   ├── seasonal_dataset/                     # Сезонные изображения
#### │   ├── stitched_seasons_austin/              # Сшитые сезонные карты
#### │   ├── test_dataset/                          # Тестовые датасеты
#### │   ├── seasonal_csv/                          # CSV сезонных данных
#### │   ├── chicago/                               # Данные Чикаго
#### │   │   ├── maps/
#### │   │   ├── patches/
#### │   │   └── data/
#### │   └── rotated_patches/                       # Повернутые патчи
#### │
#### ├── README.md                               # Описание проекта
#### ├── requirements.txt                         # Зависимости
#### ├── .gitignore                                # Gitignore файл
#### └── main.py                                   # ОСНОВНОЙ СКРИПТ ЗАПУСКА