GeoAnchor/                                  
│
├── config.py                              # Конфигурационный файл
│
├── inria_aerial_dataset/                  # Исходный датасет
│   ├── AerialImageDataset/
│   │   ├── train/
│   │   │   ├── images/                    # Оригинальные TIFF (5000x5000)
│   │   │   │   ├── austin1.tif
│   │   │   │   ├── austin2.tif
│   │   │   │   ├── ...
│   │   │   │   └── vienna9.tif
│   │   │   └── gt/                        # Маски (ground truth)
│   │   │       ├── austin1.tif
│   │   │       └── ...
│   │   └── test/
│   │       ├── images/
│   │       │   ├── austin10.tif
│   │       │   └── ...
│   │       └── gt/
│   │
│   └── AerialImageDataset_for_DINOv2/     # Создается автоматически
│       ├── processed_images/              # Обработанные изображения (518x518)
│       │   ├── austin1_processed.tiff
│       │   ├── austin2_processed.tiff
│       │   └── ...
│
├── src/                                   # Исходный код
│   ├── __init__.py                        
│   │
│   ├── data/                              # Модуль для работы с данными
│   │   ├── __init__.py
│   │   ├── preprocess_images.py          
│   │
│   ├── models/                            # Модуль для моделей
│   │
│   ├── visualization/                     # Модуль для визуализации
│   │
│   └── utils/                             # Вспомогательные функции
│
├── README.md                              # Описание проекта
├── .gitignore                             # Git ignore файл
└── main.py                                # ОСНОВНОЙ СКРИПТ ЗАПУСКА