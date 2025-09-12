import os
import io
import re
import json
import uuid
import traceback
from datetime import datetime
from typing import List, Optional
from abc import ABC, abstractmethod

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract

# ------------------- CONFIGURAÇÃO -------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DATABASE_URL = "postgresql://user:123456@localhost/nfse_db?client_encoding=utf8"

BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

# ------------------- MODELOS DE BANCO -------------------
class Task(Base):
    __tablename__ = "task"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="pendente")
    data_criacao = Column(DateTime, default=datetime.utcnow)
    data_conclusao = Column(DateTime, nullable=True)
    arquivo_pdf = Column(String, nullable=True)
    json_resultado = Column(Text, nullable=True)
    erro_mensagem = Column(String, nullable=True)

class Webhook(Base):
    __tablename__ = "webhook"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String)
    data_criacao = Column(DateTime, default=datetime.utcnow)
    actions = Column(String)

Base.metadata.create_all(bind=engine, checkfirst=True)

# ------------------- ESQUEMAS -------------------
class WebhookCreate(BaseModel):
    url: str
    actions: str

class WebhookResponse(BaseModel):
    id: int
    url: str
    data_criacao: datetime
    actions: str

class TaskResponse(BaseModel):
    id: int
    status: str
    data_criacao: datetime
    data_conclusao: Optional[datetime] = None
    arquivo_pdf: Optional[str] = None
    erro_mensagem: Optional[str] = None

# ------------------- EXTRACTOR -------------------
class NFSeExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> dict:
        pass

class FortalezaNFSeExtractor(NFSeExtractor):
    def __init__(self):
        self.template_path = 'brasao_fortaleza.png'
        self.fixed_crops = [
            [306, 372, 541, 936],    # data_emissao
            [83, 303, 1985, 2395],   # numero_nfse
            [511, 574, 825, 2396],   # prestador_nome
            [643, 702, 695, 1110],   # prestador_cnpj
            [699, 758, 775, 2396],   # prestador_endereco
            [904, 946, 517, 2392],   # tomador_nome
            [970, 1012, 328, 713],   # tomador_cnpj
            [1029, 1071, 411, 2392], # tomador_endereço
            [1224, 1857, 88, 2392],  # servicos_descrição
            [2445, 2500, 584, 909],  # valor_servicos
            [2925, 3029, 2027, 2381],# valor_iss
            [2932, 3022, 584, 907],  # valor_liquido
            [2514, 2584, 2019, 2393] # valor_deducoes
        ]

    def pdf_to_image(self, pdf_path: str, page_num: int = 0) -> np.ndarray:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()
        return img

    def extract_text_from_crop(self, img: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> str:
        if y_end <= y_start or x_end <= x_start:
            return ""
        crop = img[y_start:y_end, x_start:x_end]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((1, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        config = "--oem 3 --psm 6"
        try:
            return pytesseract.image_to_string(thresh, lang="por", config=config).strip()
        except pytesseract.TesseractError:
            return ""

    def detect_brasao(self, img: np.ndarray) -> bool:
        template = cv2.imread(self.template_path)
        if template is None:
            raise ValueError("Template 'brasao_fortaleza.png' não encontrado.")
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        template = cv2.filter2D(template, -1, kernel)

        best_val = -np.inf
        scales = np.linspace(0.5, 1.5, 20)
        for scale in scales:
            resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            if resized_template.shape[0] >= img.shape[0] or resized_template.shape[1] >= img.shape[1]:
                continue
            res = np.mean([cv2.matchTemplate(cv2.split(img)[i], cv2.split(resized_template)[i], cv2.TM_CCOEFF_NORMED) for i in range(3)], axis=0)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
            if max_val >= 0.6:
                return True
        return best_val >= 0.6

    def parse_fortaleza(self, img: np.ndarray) -> dict:
        dados = {
            "data_emissao": None,
            "numero_nfse": None,
            "prestador": {"nome": None, "cnpj": None, "endereco": None},
            "tomador": {"nome": None, "cpf_cnpj": None, "endereco": None},
            "servicos": [{"descricao": None, "quantidade": 1, "valor_unitario": None, "valor_total": None}],
            "valores": {"valor_servicos": 0.0, "valor_deducoes": 0.0, "valor_iss": 0.0, "valor_liquido": 0.0}
        }

        field_map = [
            "data_emissao", "numero_nfse", "prestador_nome", "prestador_cnpj", "prestador_endereco",
            "tomador_nome", "tomador_cnpj", "tomador_endereco", "servicos_descricao",
            "valor_servicos", "valor_iss", "valor_liquido", "valor_deducoes"
        ]

        textos = {field: self.extract_text_from_crop(img, *crop) for field, crop in zip(field_map, self.fixed_crops)}

        # Processamento com validação
        data_match = re.search(r"(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})", textos["data_emissao"])
        if data_match:
            dados["data_emissao"] = datetime.strptime(data_match.group(1), "%d/%m/%Y %H:%M:%S").isoformat()

        numero_match = re.search(r"(\d+)", textos["numero_nfse"])
        if numero_match:
            dados["numero_nfse"] = numero_match.group(1)

        dados["prestador"]["nome"] = textos["prestador_nome"].strip() or None
        cnpj_match = re.search(r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})", textos["prestador_cnpj"])
        if cnpj_match:
            dados["prestador"]["cnpj"] = cnpj_match.group(1)
        dados["prestador"]["endereco"] = textos["prestador_endereco"].strip() or None

        dados["tomador"]["nome"] = textos["tomador_nome"].strip() or None
        cnpj_match = re.search(r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})", textos["tomador_cnpj"])
        if cnpj_match:
            dados["tomador"]["cpf_cnpj"] = cnpj_match.group(1)
        dados["tomador"]["endereco"] = textos["tomador_endereco"].strip() or None

        dados["servicos"][0]["descricao"] = textos["servicos_descricao"].strip() or None

        for field in ["valor_servicos", "valor_iss", "valor_liquido", "valor_deducoes"]:
            match = re.search(r"([\d\.,]+)", textos[field])
            if match and match.group(1).replace('.', '').replace(',', '').isdigit():
                dados["valores"][field] = float(match.group(1).replace('.', '').replace(',', '.'))
            else:
                dados["valores"][field] = 0.0

        return dados

    def extract(self, file_path: str) -> dict:
        img = self.pdf_to_image(file_path)
        if self.detect_brasao(img):
            return self.parse_fortaleza(img)
        return {"error": "Brasão não detectado. Pode não ser uma NFS-e de Fortaleza."}

# Instância do extrator
extractor = FortalezaNFSeExtractor()

# ------------------- HELPERS -------------------
def safe_save_upload(upload: UploadFile, dest_folder: str) -> str:
    filename = os.path.basename(upload.filename)
    unique = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}_{filename}"
    file_path = os.path.join(dest_folder, unique)
    with open(file_path, "wb") as f:
        f.write(upload.file.read())
    return file_path

# ------------------- WEBHOOK NOTIFY -------------------
def notify_webhooks(action: str, task_id: int):
    db = Session()
    try:
        webhooks = db.query(Webhook).filter(Webhook.actions.contains(action)).all()
        for wh in webhooks:
            try:
                requests.post(wh.url, json={"action": action, "task_id": task_id, "timestamp": str(datetime.utcnow())}, timeout=5)
            except Exception:
                continue
    finally:
        db.close()

# ------------------- BACKGROUND PROCESS -------------------
def process_nfse(task_id: int):
    db = Session()
    task = None
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return
        task.status = "em andamento"
        db.commit()

        try:
            extracted = extractor.extract(task.arquivo_pdf)
            task.json_resultado = json.dumps(extracted, ensure_ascii=False)
            task.status = "concluída"
            task.data_conclusao = datetime.utcnow()
        except Exception as e:
            task.status = "falha"
            task.erro_mensagem = f"{str(e)}\n{traceback.format_exc()}"
        db.commit()
        notify_webhooks("conclusao", task_id)
    finally:
        db.close()
        if task and task.arquivo_pdf and os.path.exists(task.arquivo_pdf):
            os.remove(task.arquivo_pdf)

# ------------------- ENDPOINTS -------------------
app = FastAPI(title="Leitor de NFSe API")

@app.get("/")
def root():
    return {"message": "API de Leitura de NFSe rodando!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/upload-nfse")
def upload_info():
    return {"message": "Use POST /upload-nfse com multipart/form-data: file=@arquivo.pdf"}

@app.post("/upload-nfse")
def upload_nfse(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    db = Session()
    try:
        file_path = safe_save_upload(file, TEMP_DIR)
        task = Task(status="pendente", arquivo_pdf=file_path, data_criacao=datetime.utcnow())
        db.add(task)
        db.commit()
        db.refresh(task)
        background_tasks.add_task(process_nfse, task.id)
        notify_webhooks("upload", task.id)
        return {"task_id": task.id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})
    finally:
        db.close()

@app.get("/status/{task_id}")
def get_status(task_id: int = Path(...)):
    db = Session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task_id": task.id, "status": task.status, "data_criacao": task.data_criacao, "data_conclusao": task.data_conclusao}
    finally:
        db.close()

@app.get("/result/{task_id}")
def get_result(task_id: int = Path(...)):
    db = Session()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        if task.json_resultado:
            return json.loads(task.json_resultado)
        raise HTTPException(status_code=400, detail="No result available yet")
    finally:
        db.close()

@app.post("/webhook", response_model=WebhookResponse)
def create_webhook(webhook: WebhookCreate):
    db = Session()
    try:
        db_webhook = Webhook(url=webhook.url, actions=webhook.actions)
        db.add(db_webhook)
        db.commit()
        db.refresh(db_webhook)
        return db_webhook
    finally:
        db.close()

@app.get("/webhook", response_model=List[WebhookResponse])
def list_webhooks():
    db = Session()
    try:
        webhooks = db.query(Webhook).all()
        return webhooks
    finally:
        db.close()

@app.get("/admin/tasks", response_model=List[TaskResponse])
def admin_list_tasks(limit: int = Query(100, gt=0, le=1000), status: Optional[str] = Query(None)):
    db = Session()
    try:
        q = db.query(Task).order_by(Task.id.desc())
        if status:
            q = q.filter(Task.status == status)
        tasks = q.limit(limit).all()
        return [TaskResponse(id=t.id, status=t.status, data_criacao=t.data_criacao, data_conclusao=t.data_conclusao, arquivo_pdf=t.arquivo_pdf, erro_mensagem=t.erro_mensagem) for t in tasks]
    finally:
        db.close()

@app.get("/admin/webhooks", response_model=List[WebhookResponse])
def admin_list_webhooks(limit: int = Query(100, gt=0, le=1000)):
    db = Session()
    try:
        webhooks = db.query(Webhook).order_by(Webhook.id.desc()).limit(limit).all()
        return webhooks
    finally:
        db.close()