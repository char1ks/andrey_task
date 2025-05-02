import requests
import json
from PIL import Image
import base64
from io import BytesIO


class LLMDefectDetector:
    def __init__(self, lm_studio_url="http://localhost:1234/v1"):
        self.api_url = lm_studio_url
        self.headers = {"Content-Type": "application/json"}

        # Системный промпт для точного детектирования дефектов
        self.system_prompt = """Ты - специалист по контролю качества. 
Анализируй изображения материалов и находи дефекты. Отвечай в формате:
1.Предмет/Материал
2. Тип дефекта: [название]
3. Локация: [x,y,w,h]
4. Уверенность: [0-100%]
5. Описание: [текст]
"""

    def _encode_image(self, image_path):
        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:
            img = image_path

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def detect_defects(self, image_path, prompt=None):
        prompt = "Я передал тебе изображение,твоя задача-определить материал/предмет и найти возможные дефекты ,если таков дефектов попросту нет,то возвращай,как ответ 'Нет'.Изображение приближены очень сильно,как дефект можно воспринимать потертость в тексте,небольшие сколы и прочее"

        base64_image = self._encode_image(image_path)

        payload = {
            "model": "llava",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing response: {str(e)}")

    def _parse_response(self, response):
        try:
            content = response['choices'][0]['message']['content']

            # Парсинг структурированного ответа
            defects = []
            current_defect = {}

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('1. Тип дефекта:'):
                    if current_defect:
                        defects.append(current_defect)
                    current_defect = {
                        'type': line.split(':', 1)[1].strip(),
                        'location': None,
                        'confidence': None,
                        'description': None
                    }
                elif line.startswith('2. Локация:'):
                    loc_str = line.split(':', 1)[1].strip()
                    try:
                        current_defect['location'] = [int(x) for x in loc_str.strip('[]').split(',')]
                    except:
                        current_defect['location'] = loc_str
                elif line.startswith('3. Уверенность:'):
                    conf_str = line.split(':', 1)[1].strip().rstrip('%')
                    try:
                        current_defect['confidence'] = float(conf_str)
                    except:
                        current_defect['confidence'] = conf_str
                elif line.startswith('4. Описание:'):
                    current_defect['description'] = line.split(':', 1)[1].strip()

            if current_defect:
                defects.append(current_defect)

            return {
                'defects': defects,
                'raw_response': content
            }

        except Exception as e:
            return {
                'error': str(e),
                'raw_response': response
            }


# Пример использования с обработкой ошибок
if __name__ == "__main__":
    try:
        detector = LLMDefectDetector()

        # Анализ изображения
        result = detector.detect_defects(
            image_path="картиночки/test/img/carpet_metal_contamination_015.png",
            prompt="""Я передал тебе изображение,твоя задача-определить материал/предмет и найти возможные дефекты ,если таков дефектов попросту нет,то возвращай,как ответ 'Нет'"""
        )

        if 'error' in result:
            print(f"Ошибка: {result['error']}")
        else:
            print("Обнаруженные дефекты:")
            for i, defect in enumerate(result['defects'], 1):
                print(f"\nДефект {i}:")
                print(f"Тип: {defect.get('type', 'не указан')}")
                print(f"Локация: {defect.get('location', 'не указана')}")
                print(f"Уверенность: {defect.get('confidence', 'не указана')}%")
                print(f"Описание: {defect.get('description', 'нет описания')}")

            print("\nПолный ответ модели:")
            print(result['raw_response'])

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")