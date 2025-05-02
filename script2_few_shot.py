import torch
import clip
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import urllib.request
import os
import glob
from sklearn.model_selection import train_test_split

from torch import nn
from tqdm import tqdm
import warnings
import timm
from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

import os
import glob
import torch
import clip
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoImageProcessor, AutoModel


class MVTecDataset(Dataset):
    def __init__(self, image_paths, image_processor, clip_model, device, product_categories):
        self.image_paths = []
        # Фильтрация путей к изображениям
        for path in image_paths:
            try:
                filename = os.path.basename(path)
                parts = filename.split('_')
                if len(parts) >= 2:  # Проверка формата имени файла
                    self.image_paths.append(path)
            except:
                continue

        self.image_processor = image_processor
        self.clip_model = clip_model
        self.device = device
        self.product_categories = product_categories
        self._create_label_mapping()

    def _create_label_mapping(self):
        self.label_map = {}
        self.classes = []

        # Собираем уникальные классы
        for path in self.image_paths:
            filename = os.path.basename(path)
            parts = filename.split('_')
            category = parts[0]
            defect = 'good' if parts[1] == 'good' else '_'.join(parts[1:-1])
            class_name = f"{category}_{defect}"

            if class_name not in self.classes:
                self.classes.append(class_name)

        # Создаем mapping с фиксированными индексами
        self.classes = sorted(self.classes)  # Сортируем для воспроизводимости
        self.label_map = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)
        parts = filename.split('_')

        # Обработка имени файла
        if len(parts) < 2:
            category = 'unknown'
            defect = 'unknown'
        else:
            category = parts[0]
            defect = 'good' if parts[1] == 'good' else '_'.join(parts[1:-1])

        # Загрузка изображения с обработкой ошибок
        try:
            image = Image.open(path).convert('RGB')
        except:
            # Возвращаем нули в случае ошибки
            dummy_image = Image.new('RGB', (224, 224))
            inputs = self.image_processor(images=dummy_image, return_tensors="pt").to(self.device)
            return {
                'pixel_values': inputs['pixel_values'].squeeze(),
                'text_features': torch.zeros(512, device=self.device),
                'label': torch.tensor(-1, dtype=torch.long, device=self.device)
            }

        # Генерация текстового описания
        if defect == 'good':
            desc = self.product_categories.get(category, {}).get("description", category)
            text = f"нормальный {desc}"
        else:
            desc = self.product_categories.get(category, {}).get("description", category)
            if defect in ["маркировка", "marking"] and category in ["pill", "capsule"]:
                text = f"{desc} с плохой маркировкой"
            elif defect in ["отверстие", "hole"] and category == "hazelnut":
                text = f"{desc} с отверстием"
            elif defect in ["загрязнение", "contamination"]:
                text = f"грязный {desc}"
            else:
                text = f"{desc} с {defect}"

        # Получение текстовых эмбеддингов
        text_input = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input).squeeze()

        # Обработка изображения
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs['pixel_values'].squeeze()

        # Получение метки класса
        class_name = f"{category}_{defect}"
        label = self.label_map.get(class_name, -1)  # -1 для неизвестных классов

        return {
            'pixel_values': pixel_values,
            'text_features': text_features,
            'label': torch.tensor(label, dtype=torch.long, device=self.device)
        }


class MVTecDefectClassifier:
    def __init__(self, model_name="facebook/convnextv2-large-1k-224", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Инициализация обработчика изображений
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Инициализация модели изображений
        self.image_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.image_feature_dim = self._get_image_feature_dim()

        # Инициализация CLIP для текста
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        self.text_feature_dim = 512  # Для CLIP ViT-B/32

        # Комбинированная размерность
        self.combined_feature_dim = self.image_feature_dim + self.text_feature_dim

        # Категории продуктов
        self.product_categories = {
            # ... (ваш словарь категорий остается без изменений)
        }

        # Инициализация классификатора
        self.classifier_head = nn.Sequential(
            nn.Linear(self.combined_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        ).to(self.device)

        # Временный классификатор (будет заменен при обучении)
        self.final_classifier = nn.Linear(256, 1).to(self.device)

        # Оптимизатор и функция потерь
        self.criterion = CrossEntropyLoss()
        self.optimizer = None

        # Информация о классах
        self.num_classes = None
        self.class_names = None

    def _get_image_feature_dim(self):
        """Определяет размерность фичей изображения"""
        if hasattr(self.image_model.config, 'hidden_size'):
            return self.image_model.config.hidden_size
        elif hasattr(self.image_model.config, 'hidden_sizes'):
            return self.image_model.config.hidden_sizes[-1]
        else:
            return 1024  # Значение по умолчанию

    def train(self, train_dir, epochs=5, batch_size=8, val_split=0.2):
        # Сбор путей к изображениям
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(train_dir, "*", ext)))
            image_paths.extend(glob.glob(os.path.join(train_dir, ext)))

        if not image_paths:
            raise ValueError(f"No images found in {train_dir}")

        # Разделение на train/validation
        train_paths, val_paths = train_test_split(image_paths, test_size=val_split, random_state=42)

        # Создание датасетов
        train_dataset = MVTecDataset(train_paths, self.image_processor, self.clip_model,
                                     self.device, self.product_categories)
        val_dataset = MVTecDataset(val_paths, self.image_processor, self.clip_model,
                                   self.device, self.product_categories)

        # Обновление классификатора
        self.num_classes = train_dataset.num_classes
        self.class_names = train_dataset.classes
        self.final_classifier = nn.Linear(256, self.num_classes).to(self.device)

        # Инициализация оптимизатора
        self.optimizer = AdamW([
            {'params': self.image_model.parameters(), 'lr': 1e-5},
            {'params': self.classifier_head.parameters(), 'lr': 1e-4},
            {'params': self.final_classifier.parameters(), 'lr': 1e-4}
        ])

        # DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)

        # Цикл обучения
        best_val_acc = 0.0
        for epoch in range(epochs):
            self.image_model.train()
            self.classifier_head.train()
            self.final_classifier.train()

            train_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                pixel_values = batch['pixel_values'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                labels = batch['label'].to(self.device)

                # Прямой проход
                outputs = self.image_model(pixel_values=pixel_values)

                # Получение фичей изображения
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                else:
                    image_features = outputs.last_hidden_state.mean(dim=1)

                # Комбинирование фичей
                combined_features = torch.cat([image_features, text_features], dim=1)

                # Классификация
                features = self.classifier_head(combined_features)
                logits = self.final_classifier(features)

                # Вычисление потерь
                loss = self.criterion(logits, labels)

                # Обратный проход
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Статистика
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Валидация
            val_loss, val_acc = self._validate(val_loader)

            # Вывод статистики
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {100 * correct / total:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {100 * val_acc:.2f}%")

            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_model.pth")
                print("Best model saved!")

    def _validate(self, val_loader):
        self.image_model.eval()
        self.classifier_head.eval()
        self.final_classifier.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                text_features = batch['text_features'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.image_model(pixel_values=pixel_values)

                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                else:
                    image_features = outputs.last_hidden_state.mean(dim=1)

                combined_features = torch.cat([image_features, text_features], dim=1)
                features = self.classifier_head(combined_features)
                logits = self.final_classifier(features)

                loss = self.criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return val_loss / len(val_loader), correct / total

    def save_model(self, path):
        torch.save({
            'image_model_state_dict': self.image_model.state_dict(),
            'classifier_head_state_dict': self.classifier_head.state_dict(),
            'final_classifier_state_dict': self.final_classifier.state_dict(),
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'model_name': self.model_name,
            'image_processor_config': self.image_processor.__dict__
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.image_model.load_state_dict(checkpoint['image_model_state_dict'])
        self.classifier_head.load_state_dict(checkpoint['classifier_head_state_dict'])
        self.final_classifier.load_state_dict(checkpoint['final_classifier_state_dict'])
        self.class_names = checkpoint['class_names']
        self.num_classes = checkpoint['num_classes']
        self.model_name = checkpoint.get('model_name', "facebook/convnextv2-large-1k-224")

        # Восстановление конфигурации image_processor
        for k, v in checkpoint['image_processor_config'].items():
            setattr(self.image_processor, k, v)

        print(f"Model loaded from {path} with {self.num_classes} classes")


class SAMDefectSegmenter:
    def __init__(self, model_type="vit_h", checkpoint_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        if checkpoint_path is None:
            checkpoint_path = self.download_sam_weights(model_type)
        elif not os.path.exists(checkpoint_path):
            warnings.warn(f"Checkpoint file not found at {checkpoint_path}, downloading...")
            checkpoint_path = self.download_sam_weights(model_type)

        self._validate_checkpoint(checkpoint_path)
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)

        # Улучшенные параметры для более точной сегментации
        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=64,
            pred_iou_thresh=0.95,
            stability_score_thresh=0.96,
            crop_n_layers=2,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=200,
            output_mode="binary_mask"
        )

    def _validate_checkpoint(self, checkpoint_path):
        try:
            torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            warnings.warn(f"Checkpoint file is corrupted: {str(e)}, re-downloading...")
            os.remove(checkpoint_path)
            checkpoint_path = self.download_sam_weights(self.model_type)
        return checkpoint_path

    @staticmethod
    def download_sam_weights(model_type="vit_h"):
        weights_dir = "sam_weights"
        os.makedirs(weights_dir, exist_ok=True)

        model_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }

        save_path = os.path.join(weights_dir, f"sam_{model_type}.pth")

        if not os.path.exists(save_path):
            print(f"Downloading SAM {model_type} weights...")
            try:
                with urllib.request.urlopen(model_urls[model_type]) as response, \
                        open(save_path, 'wb') as out_file, \
                        tqdm(
                            unit='B', unit_scale=True, unit_divisor=1024,
                            miniters=1, desc=save_path,
                            total=int(response.info().get('Content-Length', 0))
                        ) as progress:

                    while True:
                        chunk = response.read(16 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        progress.update(len(chunk))

                print(f"Weights successfully downloaded to {save_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download weights: {str(e)}")

        return save_path

    def segment(self, image, defect_bbox=None):
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Конвертируем в RGB, если изображение одноканальное
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]  # Отбрасываем альфа-канал

        if defect_bbox is not None:
            x, y, w, h = defect_bbox
            x, y = max(0, x), max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            if w <= 0 or h <= 0:
                return []

            cropped_image = image[y:y + h, x:x + w]
            masks = self.mask_generator.generate(cropped_image)

            for mask in masks:
                orig_bbox = mask['bbox']
                mask['bbox'] = (
                    int(orig_bbox[0] + x),
                    int(orig_bbox[1] + y),
                    int(orig_bbox[2]),
                    int(orig_bbox[3])
                )

                seg = mask['segmentation']
                new_seg = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                new_seg[y:y + h, x:x + w] = seg
                mask['segmentation'] = new_seg
                mask['area'] = new_seg.sum()
        else:
            masks = self.mask_generator.generate(image)

        # Дополнительное уточнение масок
        refined_masks = []
        for mask in masks:
            refined = self._refine_mask(mask)
            refined_masks.append(refined)

        return refined_masks

    def _refine_mask(self, mask):
        # Морфологическое уточнение маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(mask['segmentation'].astype(np.uint8),
                                   cv2.MORPH_CLOSE, kernel)
        mask['segmentation'] = refined.astype(bool)
        return mask

    def filter_masks(self, masks, min_area_ratio=0.001, max_area_ratio=0.9, image_shape=None):
        if not masks:
            return []

        if image_shape is None:
            h, w = masks[0]['segmentation'].shape
        else:
            h, w = image_shape

        total_pixels = h * w
        min_area = min_area_ratio * total_pixels
        max_area = max_area_ratio * total_pixels

        return [mask for mask in masks if min_area <= mask.get('area', mask['segmentation'].sum()) <= max_area]

    def _cleanup(self):
        """Освобождает ресурсы модели"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DefectDetectionPipeline:
    def __init__(self, clip_model="facebook/convnextv2-large-1k-224", sam_model="vit_h", sam_checkpoint=None):
        self.classifier = MVTecDefectClassifier(clip_model)
        self.segmenter = SAMDefectSegmenter(sam_model, sam_checkpoint)

    def train_classifier(self, train_dir, epochs=5, batch_size=8, val_split=0.2):
        """Обучение классификатора на пользовательских данных"""
        self.classifier.train(train_dir, epochs=epochs, batch_size=batch_size, val_split=val_split)

    def process(self, image_path, product_category=None, defects=None, visualize=False, min_area_ratio=0.01):
        try:
            # Открываем изображение и гарантируем, что оно в формате RGB
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            # Дополнительная проверка на случай, если конвертация не сработала
            if len(image_np.shape) == 2:
                image_np = np.stack((image_np,) * 3, axis=-1)
                image = Image.fromarray(image_np)

        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

        classification = self.classifier.classify(image, product_category, defects)

        # Извлекаем предсказанную категорию и дефект
        top_prediction = classification['predictions'][0][0]
        predicted_category, predicted_defect = self._parse_prediction(top_prediction)

        masks = self.segmenter.segment(image)
        filtered_masks = self.segmenter.filter_masks(
            masks,
            min_area_ratio=min_area_ratio,
            image_shape=image_np.shape[:2]
        )

        if visualize:
            self._visualize_results(image_np, classification, filtered_masks,
                                    predicted_category, predicted_defect)

        return {
            "product_category": predicted_category,
            "defect": predicted_defect,
            "classification": classification,
            "segmentation": filtered_masks,
            "image_size": image_np.shape[:2]
        }

    def _parse_prediction(self, prediction_text):
        """Разбирает предсказание на категорию и дефект"""
        if "нормальный" in prediction_text:
            parts = prediction_text.split(" ")
            return parts[1], "good"

        parts = prediction_text.split(" с ")
        if len(parts) == 2:
            # Извлекаем категорию из описания
            category_desc = parts[0]
            for cat, data in self.classifier.product_categories.items():
                if data["description"] in category_desc:
                    return cat, parts[1]

            # Если не нашли точное совпадение, возвращаем как есть
            return parts[0], parts[1]

        return "unknown", "unknown"

    def _visualize_results(self, image, classification, masks, category, defect):
        plt.figure(figsize=(24, 12))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Original Image\nCategory: {category}\nDefect: {defect}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)

        for i, mask in enumerate(masks[:5]):  # Ограничиваем 5 самыми значимыми дефектами
            self._show_mask(mask['segmentation'], plt.gca(), random_color=True)
            self._show_box(mask['bbox'], plt.gca(), label=f"Defect {i + 1}")

        top_pred = classification['predictions'][0]
        plt.title(f"Top prediction: {top_pred[0]} ({top_pred[1]:.2f})\nDetected {len(masks)} defects")
        plt.axis('off')

        plt.show()

    def _show_mask(self, mask, ax, random_color=False):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color \
            else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _show_box(self, box, ax, label=""):
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        ax.text(x0, y0, label, color='white', fontsize=12,
                bbox=dict(facecolor='green', alpha=0.7, pad=1))

    def cleanup(self):
        """Очистка ресурсов всех моделей"""
        self.classifier._cleanup()
        self.segmenter._cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        pipeline = DefectDetectionPipeline(
            clip_model="facebook/convnextv2-large-1k-224",
            sam_model="vit_h"
        )

        # Пример обучения модели
        print("Training classifier...")
        pipeline.train_classifier(
            train_dir="картиночки/train/",
            epochs=5,
            batch_size=8,
            val_split=0.2
        )

        # Пример использования обученной модели
        print("\nTesting on validation image...")
        results = pipeline.process(
            image_path="картиночки/test/img/grid_metal_contamination_004.png",
            product_category=None,
            defects=None,
            visualize=True,
            min_area_ratio=0.005
        )

        print("\n=== Defect Detection Results ===")
        print(f"Predicted Category: {results['product_category']}")
        print(f"Predicted Defect: {results['defect']}")
        print("\nTop Predictions:")
        for i, (class_name, prob) in enumerate(results['classification']['predictions'], 1):
            print(f"{i}. {class_name}: {prob:.4f}")

        print(f"\nDetected {len(results['segmentation'])} defect regions")
        for i, defect in enumerate(results['segmentation'][:3]):
            print(f"Defect {i + 1}: Area={defect['area']}, Confidence={defect['predicted_iou']:.2f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if 'pipeline' in locals():
            pipeline.cleanup()
    finally:
        if 'pipeline' in locals():
            pipeline.cleanup()