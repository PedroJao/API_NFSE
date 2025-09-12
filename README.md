Leitor de NFSe API 📊
Uma API robusta para processar Notas Fiscais de Serviços Eletrônicas (NFSe) de Fortaleza, Brasil, com OCR e integração a PostgreSQL.
Bem-vindo ao Leitor de NFSe API, uma aplicação baseada em FastAPI projetada para extrair dados estruturados de PDFs de NFSe usando OCR, com processamento assíncrono, notificações via webhooks e gerenciamento avançado de tarefas. Otimizada para o formato de Fortaleza, esta API é modular e extensível.
📖 Visão Geral
O Leitor de NFSe API automatiza a extração de dados de NFSe em PDF, utilizando Tesseract OCR para identificar informações específicas. Os resultados são armazenados em um banco de dados PostgreSQL, com suporte a tarefas em segundo plano e notificações em tempo real via webhooks. A arquitetura segue os princípios de Clean Code e SOLID, sendo ideal para expansão.
🚀 Funcionalidades Principais

Upload e Processamento de PDFs: Envie PDFs para extração assíncrona.
Extração via OCR: Use Tesseract para extrair texto de regiões definidas.
Notificações via Webhook: Receba alertas em eventos como upload e conclusão.
Armazenamento Persistente: Salve metadados e resultados em PostgreSQL.
Gerenciamento Avançado: Monitore tarefas e webhooks via endpoints administrativos.

🛠️ Dependências
Backend













































BibliotecaDescriçãoVersão SugeridafastapiFramework web assíncrono0.100.0+sqlalchemyORM para PostgreSQL2.0.0+pymupdf (fitz)Renderização de PDFs em imagens1.22.0+opencv-pythonProcessamento de imagens4.8.0+pytesseractExtração de texto via OCR0.3.10+requestsEnvio de notificações HTTP2.28.0+pydanticValidação e serialização de dados2.4.0+
Requisitos Adicionais

Tesseract OCR: Instale e configure o caminho (ex.: C:\Program Files\Tesseract-OCR\tesseract.exe).
PostgreSQL: Configure com URL postgresql://user:123456@localhost/nfse_db.

⚙️ Configuração

Caminho do Tesseract:
pythonpytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
Ajuste o caminho conforme sua instalação.
URL do Banco de Dados:
pythonDATABASE_URL = "postgresql://user:123456@localhost/nfse_db?client_encoding=utf8"
Substitua user e 123456 por suas credenciais.
Diretório Temporário:
pythonTEMP_DIR = Path.cwd() / "temp"
TEMP_DIR.mkdir(exist_ok=True)
Cria um diretório temp para arquivos temporários.
SQLAlchemy:
pythonengine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Configura o ORM e a base declarativa.

📊 Modelos de Banco de Dados
Task













































CampoTipoDescriçãoidIntegerChave primária auto-incrementadastatusStringStatus (pendente, em andamento, concluída, falha)created_atDateTimeData de criaçãocompleted_atDateTimeData de conclusão (opcional)pdf_pathStringCaminho do PDF (opcional)result_jsonTextResultado em JSON (opcional)error_messageStringMensagem de erro (opcional)
Webhook






























CampoTipoDescriçãoidIntegerChave primária auto-incrementadaurlStringURL do webhookcreated_atDateTimeData de criaçãoactionsStringAções (ex.: upload,conclusao)
Tabelas criadas com: Base.metadata.create_all(bind=engine, checkfirst=True).
📋 Esquemas Pydantic

WebhookCreate:
pythonclass WebhookCreate(BaseModel):
    url: str
    actions: str

url: URL do webhook.
actions: Ações separadas por vírgula.


WebhookResponse:
pythonclass WebhookResponse(BaseModel):
    id: int
    url: str
    created_at: datetime
    actions: str

TaskResponse:
pythonclass TaskResponse(BaseModel):
    id: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    pdf_path: Optional[str] = None
    error_message: Optional[str] = None


🔍 Extrator de NFSe

NFSeExtractor (Abstrata):

extract(file_path: str) -> dict: Interface para extratores.


FortalezaNFSeExtractor:

Atributos: template_path (logotipo), fixed_crops (coordenadas de recorte).
Métodos:

pdf_to_image: Converte PDF em imagem (300 DPI).
extract_text_from_crop: Extrai texto com OCR.
detect_template: Detecta logotipo (limiar 0.6).
parse_fortaleza: Extrai dados estruturados.
extract: Combina detecção e parsing.





🛠️ Funções Auxiliares

safe_save_upload(upload: UploadFile, dest_folder: str) -> str:

Salva arquivos com nome único em dest_folder.



🔔 Notificação via Webhook

notify_webhooks(action: str, task_id: int):

Envia {"action": "upload/conclusao", "task_id": int, "timestamp": str}.
Ignora falhas individuais.



⏳ Processamento em Segundo Plano

process_nfse(task_id: int):

Processa PDF, atualiza status, armazena resultados e notifica.



🌐 Endpoints da API







































































MétodoEndpointDescriçãoResposta/ParâmetrosGET/Confirma API ativa{"message": "API de Leitura de NFSe rodando!"}GET/healthVerifica saúde{"status": "healthy"}GET/upload-nfseInstruções de upload{"message": "Use POST /upload-nfse..."}POST/upload-nfseEnvia PDF para processamento{"task_id": int} (multipart/form-data)GET/status/{task_id}Status da tarefa{"task_id": int, "status": str, ...}GET/result/{task_id}Dados extraídosJSON com dadosPOST/webhookRegistra webhookWebhookResponse (JSON no corpo)GET/webhookLista webhooksList[WebhookResponse]GET/admin/tasksLista tarefas (limit, status)List[TaskResponse]GET/admin/webhooksLista webhooks (limit)List[WebhookResponse]
🚀 Uso
Instalação

Clone o repositório:
bashgit clone https://github.com/seu-usuario/leitor-nfse-api.git
cd leitor-nfse-api/source

Instale dependências:
bashpip install fastapi uvicorn sqlalchemy psycopg2-binary pymupdf opencv-python pytesseract requests

Configure Tesseract:

Baixe em Tesseract Wiki.
Ajuste o caminho em models/database.py.


Configure PostgreSQL:

Atualize DATABASE_URL com suas credenciais.


Adicione brasao_fortaleza.png na raiz (source/).

Execução
bashuvicorn main:app --reload
Acesse em http://localhost:8000.
Exemplos

Enviar NFSe:
bashcurl -X POST -F "file=@nfse.pdf" http://localhost:8000/upload-nfse
Resposta: {"task_id": 1}
Verificar Status:
bashcurl http://localhost:8000/status/1

Registrar Webhook:
bashcurl -X POST -H "Content-Type: application/json" -d '{"url":"https://example.com/webhook","actions":"upload,conclusao"}' http://localhost:8000/webhook


⚠️ Tratamento de Erros

Upload Falhou: 500 com detalhes.
Tarefa Não Encontrada: 404.
Resultado Indisponível: 400.
Erros de OCR: Registrados em error_message.
Falhas de Webhook: Ignoradas.

🌱 Extensibilidade

Novos Formatos: Crie classes derivadas de NFSeExtractor.
Webhooks: Adicione retentativas ou filas (ex.: RabbitMQ).
Banco: Índices e paginação.
Imagens: Melhore detecção e suporte a rotações.

🤝 Contribuição

Faça um fork.
Crie uma branch: git checkout -b feature/nova-funcionalidade.
Commit: git commit -m "Descrição".
Envie um PR com detalhes.
