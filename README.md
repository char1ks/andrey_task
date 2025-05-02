Файл script2 выглядит более многообещающим,чем остальные,рекомендую его глянуть
Комбинированный пайплайн для обнаружения и сегментации дефектов на промышленных изображениях, использующий:

CLIP для классификации дефектов

Segment Anything Model (SAM) для точной сегментации

Основные компоненты
1. CLIPDefectClassifier
Классифицирует дефекты на изображениях с помощью модели CLIP.

Методы:

classify(image, materials=None, defects=None, top_k=1):

image: PIL.Image или np.ndarray

materials: Список материалов (по умолчанию 20+ промышленных материалов)

defects: Список дефектов (по умолчанию 25+ типов дефектов)

top_k: Количество возвращаемых топ-предсказаний

Возвращает:

python
{
    "predictions": [(класс, вероятность), ...],
    "features": {
        "image": features-вектор,
        "text": текстовые features
    }
}
2. SAMDefectSegmenter
Сегментирует дефекты с помощью модели SAM.

Методы:

segment(image, defect_bbox=None):

image: PIL.Image или np.ndarray

defect_bbox: Опциональный bounding box (x,y,w,h) для фокусировки на области

filter_masks(masks, min_area_ratio=0.001, max_area_ratio=0.9): Фильтрация масок по размеру

3. DefectDetectionPipeline
Объединяет классификацию и сегментацию.

Методы:

process(image_path, defect_classes, visualize=False, min_area_ratio=0.01):

Полный цикл обработки изображения

visualize=True показывает результаты

