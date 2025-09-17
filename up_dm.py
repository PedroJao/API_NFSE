import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import re
import json
from datetime import datetime

# Converter PDF para imagem com alta qualidade (dpi=300)
def pdf_to_image(pdf_path, page_num=0):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=300)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img

# Função pra pré-processar e extrair texto de um crop
def extract_text_from_crop(img, y_start, y_end, x_start, x_end):
    if y_end <= y_start or x_end <= x_start:
        print(f"Crop inválido [{y_start}:{y_end}, {x_start}:{x_end}] - pulando.")
        return ""
    crop = img[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    config = "--oem 3 --psm 6"
    try:
        texto = pytesseract.image_to_string(thresh, lang="por", config=config).strip()
        print(f"Texto extraído do crop [{y_start}:{y_end}, {x_start}:{x_end}]: {texto[:200]}...")
        return texto
    except pytesseract.TesseractError as e:
        print(f"Erro no Tesseract: {e}")
        print("Dica: Baixe por.traineddata e set TESSDATA_PREFIX!")
        return ""

# Função pra detectar o brasão
def detect_brasao(img, template_path='brasao_fortaleza.png', threshold=0.6):
    template = cv2.imread(template_path)
    if template is None:
        raise ValueError("Template não encontrado.")
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    template = cv2.filter2D(template, -1, kernel)

    found = False
    best_val = -np.inf
    best_scale = 1.0
    scales = np.linspace(0.5, 1.5, 20)

    for scale in scales:
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        if resized_template.shape[0] >= img.shape[0] or resized_template.shape[1] >= img.shape[1]:
            continue
        res = np.mean([cv2.matchTemplate(cv2.split(img)[i], cv2.split(resized_template)[i], cv2.TM_CCOEFF_NORMED) for i in range(3)], axis=0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_scale = scale
        if max_val >= threshold:
            found = True
            break

    print(f"Melhor valor de match encontrado: {best_val:.4f} (escala: {best_scale:.2f})")

    if best_val >= threshold:
        debug_img = img.copy()
        h, w = template.shape[:2]
        resized_h = int(h * best_scale)
        resized_w = int(w * best_scale)
        top_left = max_loc
        bottom_right = (top_left[0] + resized_w, top_left[1] + resized_h)
        cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 3)
        cv2.imwrite('debug_match.png', debug_img)
        print("Debug salvo como 'debug_match.png' - quadradinho no brasão!")

    return found

# Parser com crops
def parser_fortaleza(img, crops):
    dados = {
        "data_emissao": None,
        "numero_nfse": None,
        "prestador": {"nome": None, "cnpj": None, "endereco": None},
        "tomador": {"nome": None, "cpf_cnpj": None, "endereco": None},
        "servicos": [
            {
                "descricao": None,
                "quantidade": 1,
                "valor_unitario": 0.0,
                "valor_total": 0.0
            }
        ],
        "valores": {
            "valor_servicos": 0.0,
            "valor_deducoes": 0.0,
            "valor_iss": 0.0,
            "valor_liquido": 0.0
        }
    }

    field_map = [
        "data_emissao", "numero_nfse", "prestador_nome", "prestador_cnpj", "prestador_endereco",
        "tomador_nome", "tomador_cnpj", "tomador_endereco", "servicos_descricao",
        "valor_servicos", "valor_iss", "valor_liquido", "valor_deducoes"
    ]

    textos = {}
    for i, (y_start, y_end, x_start, x_end) in enumerate(crops):
        if i < len(field_map):
            texto = extract_text_from_crop(img, y_start, y_end, x_start, x_end)
            textos[field_map[i]] = texto

    # Processar dados com validação
    data_match = re.search(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})", textos.get("data_emissao", ""))
    if data_match:
        dados["data_emissao"] = datetime.strptime(data_match.group(1), "%d/%m/%Y %H:%M:%S").isoformat()

    numero_match = re.search(r"(\d+)", textos.get("numero_nfse", ""))
    if numero_match:
        dados["numero_nfse"] = numero_match.group(1)

    dados["prestador"]["nome"] = textos.get("prestador_nome", "").strip() or None
    cnpj_match = re.search(r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})", textos.get("prestador_cnpj", ""))
    if cnpj_match:
        dados["prestador"]["cnpj"] = cnpj_match.group(1)
    dados["prestador"]["endereco"] = textos.get("prestador_endereco", "").strip() or None

    dados["tomador"]["nome"] = textos.get("tomador_nome", "").strip() or None
    cnpj_match = re.search(r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})", textos.get("tomador_cnpj", ""))
    if cnpj_match:
        dados["tomador"]["cpf_cnpj"] = cnpj_match.group(1)
    dados["tomador"]["endereco"] = textos.get("tomador_endereco", "").strip() or None

    dados["servicos"][0]["descricao"] = textos.get("servicos_descricao", "").strip() or None

    for field in ["valor_servicos", "valor_iss", "valor_liquido", "valor_deducoes"]:
        match = re.search(r"([\d\.,]+)", textos.get(field, ""))
        if match and match.group(1).replace('.', '').replace(',', '').isdigit():
            value = float(match.group(1).replace('.', '').replace(',', '.'))
            dados["valores"][field] = value
        else:
            print(f"Valor inválido para {field}: {textos.get(field, 'vazio')}. Usando None.")
            dados["valores"][field] = 0.0

    return dados


# Interface interativa pra marcar áreas
def mark_areas(img):
    img_display = img.copy()
    drawing = False
    ix, iy = -1, -1
    crops = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, crops
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_temp = img_display.copy()
                cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Marque as Áreas', img_temp)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x_end, y_end = x, y
            crop = (min(iy, y_end), max(iy, y_end), min(ix, x_end), max(ix, x_end))
            crops.append(crop)
            cv2.rectangle(img_display, (ix, iy), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow('Marque as Áreas', img_display)
            print(f"Área marcada: {crop} - Pressione 'r' pra resetar último, 's' pra salvar.")

    cv2.namedWindow('Marque as Áreas', cv2.WINDOW_NORMAL)  # Janela ajustável
    cv2.setMouseCallback('Marque as Áreas', mouse_callback)

    print("Instruções:")
    print("- Clique e arraste pra marcar uma área (quadrados verdes).")
    print("- Pressione 'r' pra resetar o último quadrado.")
    print("- Pressione 's' pra salvar e processar o JSON.")
    print("- Pressione 'q' pra sair sem salvar.")
    print("Marque na ordem: data, número, prestador_nome, cnpj, endereço, tomador_nome, cnpj, endereço, serviços, valor_servicos, iss, líquido, deduções.")

    field_map = [
        "data_emissao", "numero_nfse", "prestador_nome", "prestador_cnpj", "prestador_endereco",
        "tomador_nome", "tomador_cnpj", "tomador_endereco", "servicos_descricao",
        "valor_servicos", "valor_iss", "valor_liquido", "valor_deducoes"
    ]

    while True:
        cv2.imshow('Marque as Áreas', img_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') and crops:
            crops.pop()
            img_display = img.copy()
            for (y_start, y_end, x_start, x_end) in crops:
                cv2.rectangle(img_display, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            print("Último quadrado resetado.")
        elif key == ord('s'):
            if len(crops) >= 4:
                print(f"Salvando {len(crops)} áreas. Imprimindo coordenadas no formato solicitado:")
                print("crops = [")
                for i, crop in enumerate(crops):
                    field = field_map[i] if i < len(field_map) else f"campo_{i+1}"
                    print(f"    {list(crop)},    # {field}")
                print("]")
                print("Processando JSON...")
                cv2.destroyAllWindows()
                return crops
            else:
                print("Marque pelo menos 4 áreas antes de salvar.")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return []

    cv2.destroyAllWindows()
    return []

# ------------------- TESTE -------------------
pdf_path = "mossoro.pdf"
img = pdf_to_image(pdf_path)

if detect_brasao(img):
    print("Brasão de Fortaleza detectado. Abrindo janela pra marcar áreas...")
    crops = mark_areas(img)
    if crops:
        resultado = parser_fortaleza(img, crops)
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
        # Salva debug com quadrados
        debug_img = img.copy()
        for i, (y_start, y_end, x_start, x_end) in enumerate(crops):
            cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i), (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite('debug_marked.png', debug_img)
        print("Debug com quadrados marcados salvo como 'debug_marked.png'!")
    else:
        print("Nenhuma área marcada. Saindo.")
else:
    print("Brasão não detectado. Este pode não ser um documento de Fortaleza.")