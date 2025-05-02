import torch
import clip
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import urllib.request
import os
from tqdm import tqdm
import warnings
# Класс CLIPDefectClassifier с обновленными материалами и дефектами

class CLIPDefectClassifier:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def classify(self, image, materials=None, defects=None, top_k=1):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Расширенные материалы (популярные на заводах)
        if materials is None:
            materials = [
                "metal", "steel", "aluminum", "copper", "brass", "bronze", "iron", "stainless steel",
                "wood", "concrete", "plastic", "glass", "ceramic", "composite", "rubber", "silicone",
                "carbon fiber", "epoxy", "PVC", "fiberglass"
            ]

        # Расширенные дефекты
        if defects is None:
            defects = [
                "crack", "corrosion", "pitting", "delamination", "scratch", "dent", "discoloration",
                "deformation", "rust", "blistering", "fatigue", "peeling", "wear", "burr",
                "burn mark", "hole", "uneven surface", "stain", "warping", "chipping", "flaking",
                "contamination", "swelling", "leakage", "abrasion", "no defect"
            ]

        # Комбинированные классы
        classes = []
        for material in materials:
            for defect in defects:
                classes.append(f"{defect} on {material}")
            classes.append(f"no defect on {material}")

        # Подготовка данных
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)
            logits_per_image, _ = self.model(image_input, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[-top_k:][::-1]
        results = {
            "predictions": [(classes[i], float(probs[i])) for i in top_indices],
            "features": {
                "image": image_features.detach().cpu().numpy(),
                "text": text_features.detach().cpu().numpy()
            }
        }

        return results


class SAMDefectSegmenter:
    def __init__(self, model_type="vit_b", checkpoint_path=None, device=None):
        """
        Инициализация SAM модели для сегментации дефектов

        Args:
            model_type (str): Тип модели SAM (vit_b, vit_l, vit_h)
            checkpoint_path (str): Путь к файлу весов SAM
            device (str): Устройство для вычислений (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        if checkpoint_path is None:
            checkpoint_path = self.download_sam_weights(model_type)
        elif not os.path.exists(checkpoint_path):
            warnings.warn(f"Checkpoint file not found at {checkpoint_path}, downloading...")
            checkpoint_path = self.download_sam_weights(model_type)

        # Проверка целостности файла перед загрузкой
        self._validate_checkpoint(checkpoint_path)

        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            self.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100
        )

    def _validate_checkpoint(self, checkpoint_path):
        """Проверяет целостность файла с весами"""
        try:
            # Попытка загрузить файл, чтобы проверить его целостность
            torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            # Если файл поврежден, удаляем его и скачиваем заново
            warnings.warn(f"Checkpoint file is corrupted: {str(e)}, re-downloading...")
            os.remove(checkpoint_path)
            checkpoint_path = self.download_sam_weights(self.model_type)
        return checkpoint_path

    @staticmethod
    def download_sam_weights(model_type="vit_b"):
        """Скачивает веса SAM автоматически, если они отсутствуют"""
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
                # Используем tqdm для отображения прогресса загрузки
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
        """
        Сегментация дефектов на изображении

        Args:
            image (PIL.Image/np.ndarray): Входное изображение
            defect_bbox (tuple): Опционально - bbox дефекта (x,y,w,h)

        Returns:
            list: Список масок и связанной информации
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Конвертация изображения в RGB формат
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        if defect_bbox is not None:
            # Если задан bbox, фокусируемся на этой области
            x, y, w, h = defect_bbox
            # Проверка на выход за границы изображения
            x, y = max(0, x), max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            if w <= 0 or h <= 0:
                return []

            cropped_image = image[y:y + h, x:x + w]
            masks = self.mask_generator.generate(cropped_image)

            # Корректируем координаты масок относительно исходного изображения
            for mask in masks:
                # Обновляем bbox
                orig_bbox = mask['bbox']
                mask['bbox'] = (
                    int(orig_bbox[0] + x),
                    int(orig_bbox[1] + y),
                    int(orig_bbox[2]),
                    int(orig_bbox[3])
                )

                # Обновляем маску сегментации
                seg = mask['segmentation']
                new_seg = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                new_seg[y:y + h, x:x + w] = seg
                mask['segmentation'] = new_seg
                mask['area'] = new_seg.sum()

        else:
            masks = self.mask_generator.generate(image)

        # Конвертируем bbox в формат XYXY для единообразия
        for mask in masks:
            if 'bbox' in mask:
                mask['bbox'] = (
                    mask['bbox'][0],
                    mask['bbox'][1],
                    mask['bbox'][0] + mask['bbox'][2],
                    mask['bbox'][1] + mask['bbox'][3]
                )

        return masks

    def filter_masks(self, masks, min_area_ratio=0.001, max_area_ratio=0.9, image_shape=None):
        """
        Фильтрация масок по площади

        Args:
            masks (list): Список масок
            min_area_ratio (float): Минимальная относительная площадь маски
            max_area_ratio (float): Максимальная относительная площадь маски
            image_shape (tuple): Размеры изображения (h, w)

        Returns:
            list: Отфильтрованные маски
        """
        if not masks:
            return []

        if image_shape is None:
            # Пытаемся определить размер из первой маски
            h, w = masks[0]['segmentation'].shape
        else:
            h, w = image_shape

        total_pixels = h * w
        min_area = min_area_ratio * total_pixels
        max_area = max_area_ratio * total_pixels

        filtered = []
        for mask in masks:
            area = mask['area'] if 'area' in mask else mask['segmentation'].sum()
            if min_area <= area <= max_area:
                filtered.append(mask)
        return filtered


class DefectDetectionPipeline:
    def __init__(self, clip_model="ViT-B/32", sam_model="vit_b", sam_checkpoint=None):
        """
        Комбинированный пайплайн для обнаружения и сегментации дефектов

        Args:
            clip_model (str): Модель CLIP
            sam_model (str): Тип модели SAM
            sam_checkpoint (str): Путь к весам SAM
        """
        self.classifier = CLIPDefectClassifier(clip_model)
        self.segmenter = SAMDefectSegmenter(sam_model, sam_checkpoint)

    def process(self, image_path, defect_classes, visualize=False, min_area_ratio=0.01):
        """
        Полный процесс обработки изображения

        Args:
            image_path (str): Путь к изображению
            defect_classes (list): Список классов дефектов
            visualize (bool): Визуализировать результаты
            min_area_ratio (float): Минимальная относительная площадь маски

        Returns:
            dict: Результаты классификации и сегментации
        """
        # Загрузка изображения
        try:
            image = Image.open(image_path)
            image_np = np.array(image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

        # Классификация
        classification = self.classifier.classify(image, defect_classes)

        # Сегментация
        masks = self.segmenter.segment(image)

        # Фильтрация масок по площади
        filtered_masks = self.segmenter.filter_masks(
            masks,
            min_area_ratio=min_area_ratio,
            image_shape=image_np.shape[:2]
        )

        # Визуализация
        if visualize:
            self._visualize_results(image_np, classification, filtered_masks)

        return {
            "classification": classification,
            "segmentation": filtered_masks,
            "image_size": image_np.shape[:2]
        }

    def _visualize_results(self, image, classification, masks):
        """Визуализация результатов"""
        plt.figure(figsize=(20, 10))

        # Отображение исходного изображения
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        # Отображение сегментации
        plt.subplot(1, 2, 2)
        plt.imshow(image)

        for mask in masks:
            self._show_mask(mask['segmentation'], plt.gca(), random_color=True)
            self._show_box(mask['bbox'], plt.gca())

        title = f"Top prediction: {classification['predictions'][0][0]} ({classification['predictions'][0][1]:.2f})"
        plt.title(title)
        plt.axis('off')

        plt.show()

    def _show_mask(self, mask, ax, random_color=False):
        """Отображение маски"""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def _show_box(self, box, ax):
        """Отображение bbox"""
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":
    # Инициализация пайплайна с автоматической загрузкой весов
    try:
        pipeline = DefectDetectionPipeline(
            clip_model="ViT-L/14",  # Вместо ViT-B/32
            sam_model="vit_h",
            sam_checkpoint=None
        )
        # Классы дефектов
        defect_classes = ["crack", "corrosion", "pitting", "delamination", "no defect"]

        # Обработка изображения
        results = pipeline.process(
            image_path="картиночки/test/img/capsule_crack_021.png",
            defect_classes=defect_classes,
            visualize=True,
            min_area_ratio=0.005
        )

        # Вывод результатов
        print("Classification results:")
        for class_name, prob in results['classification']['predictions']:
            print(f"- {class_name}: {prob:.4f}")

        print(f"\nDetected {len(results['segmentation'])} defect regions")
        print(f"Image size: {results['image_size']}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Удаляем возможные поврежденные файлы весов
        if 'pipeline' in locals() and hasattr(pipeline, 'segmenter'):
            weights_path = os.path.join("sam_weights", f"sam_vit_b.pth")
            if os.path.exists(weights_path):
                try:
                    os.remove(weights_path)
                    print(f"Removed potentially corrupted weights file: {weights_path}")
                except:
                    pass