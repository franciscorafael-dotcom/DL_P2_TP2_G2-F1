from pathlib import Path
from predict import predict_image

IMAGE_PATH = Path(__file__).parent / "Esca_104.jpg" #imagem do holdout

result = predict_image(IMAGE_PATH)

print(result)