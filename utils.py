import base64
from PIL import Image
import io


def convert_image_to_base64(file):
    with Image.open(file.path) as img:
        buffer = io.BytesIO()
        img = img.convert('RGB')
        img.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_data
