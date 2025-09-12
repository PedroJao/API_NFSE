import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extrair_texto(caminho_imagem):
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    texto = pytesseract.image_to_string(thresh, lang="eng")
    return texto

if __name__ == "__main__":
    imagem = r"C:\teste_nfse\pagina1.png"
    texto_extraido = extrair_texto(imagem)
    print("Texto encontrado:\n", texto_extraido)