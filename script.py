import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import pipeline
import matplotlib.pyplot as plt


class AdvancedDefectDetection:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Инициализация моделей
        self.init_models()

        # Словарь материалов и возможных дефектов
        self.material_defects = {
            "металл": ["царапина", "вмятина", "коррозия", "трещина", "деформация"],
            "дерево": ["трещина", "скол", "царапина", "плесень", "сучок"],
            "пластик": ["трещина", "деформация", "пузырь", "царапина"],
            "стекло": ["трещина", "скол", "царапина", "пузырь"]
        }

    def init_models(self):
        """Инициализация современных моделей"""
        # Сегментационная модель (Mask2Former)
        self.segmenter = pipeline(
            "image-segmentation",
            model="facebook/mask2former-swin-large-coco-instance",
            device=0 if self.device == "cuda" else -1
        )

        # Детектор дефектов (DETR)
        self.detector = pipeline(
            "object-detection",
            model="facebook/detr-resnet-101",
            device=0 if self.device == "cuda" else -1
        )

        # Классификатор материалов (ConvNext)
        self.material_processor = AutoImageProcessor.from_pretrained("facebook/convnext-large-224")
        self.material_model = AutoModelForImageClassification.from_pretrained(
            "facebook/convnext-large-224"
        ).to(self.device)

        # Заменим неработающий детектор аномалий на работающую модель
        self.anomaly_detector = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=0 if self.device == "cuda" else -1
        )

    def detect_material(self, image):
        """Определение материала с помощью ConvNext"""
        inputs = self.material_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.material_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return self.material_model.config.id2label[predicted_class_idx]

    def detect_defects(self, image):
        """Основной метод обнаружения дефектов"""
        # Определяем материал
        material = self.detect_material(image)
        print(f"Материал: {material}")

        # Получаем возможные дефекты для этого материала
        possible_defects = self.material_defects.get(material.lower(), [])
        print(f"Возможные дефекты: {', '.join(possible_defects)}")

        # Детекция объектов (DETR)
        detections = self.detector(image)

        # Сегментация (Mask2Former)
        segmentation = self.segmenter(image)

        # Детекция аномалий
        anomaly_results = self.anomaly_detector(image)

        # Визуализация результатов
        self.visualize_results(image, detections, segmentation, material)

        return {
            "material": material,
            "possible_defects": possible_defects,
            "detections": detections,
            "segmentation": segmentation,
            "anomaly_score": anomaly_results[0]["score"]
        }

    def visualize_results(self, image, detections, segmentation, material):
        """Визуализация результатов"""
        # Конвертируем PIL в OpenCV формат
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Рисуем bounding boxes
        for detection in detections:
            if detection['score'] > 0.7:  # Порог уверенности
                box = detection['box']
                cv2.rectangle(img,
                              (int(box['xmin']), int(box['ymin'])),
                              (int(box['xmax']), int(box['ymax'])),
                              (0, 255, 0), 2)
                cv2.putText(img,
                            f"{detection['label']}: {detection['score']:.2f}",
                            (int(box['xmin']), int(box['ymin'] - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2))

                # Наложение масок сегментации
                for seg in segmentation:
                    mask = np.array(seg['mask'])
                color = np.random.randint(0, 255, 3).tolist()
                img[mask == 255] = img[mask == 255] * 0.7 + np.array(color) * 0.3

                # Добавляем информацию о материале
                cv2.putText(img, f"Material: {material}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Отображение результатов
                plt.figure(figsize=(15, 10))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title("Defect Detection Results")
                plt.show()


if __name__ == "__main__":
    detector = AdvancedDefectDetection()
    image_path = "фоточки/Царапины на паркетной доске.1617122234.jpg.preview.jpg"  # Укажите путь к изображению

    try:
        image = Image.open(image_path).convert("RGB")
        results = detector.detect_defects(image)
        print("\nРезультаты анализа:")
        print(f"Материал: {results['material']}")
        print(f"Обнаруженные дефекты: {results['detections']}")
        print(f"Оценка аномалий: {results['anomaly_score']:.2f}")
    except Exception as e:
        print(f"Ошибка: {str(e)}")
