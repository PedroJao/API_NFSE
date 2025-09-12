import fitz  # PyMuPDF

pdf_path = "nfse_geomk.pdf"
doc = fitz.open(pdf_path)

for i, pagina in enumerate(doc, start=1):
    pix = pagina.get_pixmap(dpi=300)  # aumenta DPI para melhor qualidade
    img_path = f"pagina{i}.png"
    pix.save(img_path)
    print(f"Página {i} salva como {img_path}")