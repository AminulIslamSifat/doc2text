import pytesseract as pts
from PIL import Image

image = Image.open('tests/assets/image/image.png')
data = pts.image_to_data(image, output_type=pts.Output.DICT)
with open('output.txt', 'w') as f:
    f.write(str(data))

print(pts.get_languages(config=''))
for i, text in enumerate(data["text"]):
    if text.strip():
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        conf = data["conf"][i]
        print(f"Text: {text}, Confidence: {conf}, Bounding Box: ({x}, {y}, {w}, {h})")


