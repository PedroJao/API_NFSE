Documentação da API de Processamento de NFSe
Este documento descreve uma aplicação baseada em FastAPI projetada para processar Notas Fiscais de Serviços Eletrônicas (NFSe) de Fortaleza, Brasil. A aplicação extrai dados estruturados de arquivos PDF utilizando OCR e armazena metadados de tarefas em um banco de dados PostgreSQL. Suporta processamento assíncrono de tarefas, notificações via webhooks e endpoints administrativos para gerenciamento de tarefas e webhooks.
Índice

Visão Geral
Dependências
Configuração
Modelos de Banco de Dados
Esquemas Pydantic
Extrator de NFSe
Funções Auxiliares
Notificação via Webhook
Processamento em Segundo Plano
Endpoints da API
Uso
Tratamento de Erros
Extensibilidade


Visão Geral
A aplicação processa PDFs de NFSe enviados por meio de um endpoint FastAPI, extrai dados estruturados utilizando OCR e armazena os resultados em um banco de dados PostgreSQL. Utiliza tarefas em segundo plano para processamento assíncrono, notifica webhooks registrados em eventos-chave (upload e conclusão) e fornece endpoints para verificar o status de tarefas, recuperar resultados e gerenciar webhooks. A lógica de extração é específica para o formato de NFSe de Fortaleza, utilizando regiões de recorte predefinidas e correspondência de modelo para validar documentos.
Funcionalidades principais:

Upload e Processamento de PDFs: Usuários enviam PDFs de NFSe, que são processados em segundo plano para extração de dados.
Extração via OCR: Utiliza Tesseract OCR para extrair texto de regiões específicas do PDF.
Notificações via Webhook: Notifica webhooks registrados em eventos de criação e conclusão de tarefas.
Armazenamento no Banco: Armazena metadados e resultados das tarefas em um banco de dados PostgreSQL.
Endpoints Administrativos: Permite consultar status de tarefas e webhooks.


Dependências
A aplicação depende das seguintes bibliotecas Python:

FastAPI: Framework web para construção da API.
SQLAlchemy: ORM para interações com o banco de dados PostgreSQL.
PyMuPDF (fitz): Para renderização de PDFs em imagens.
OpenCV (cv2): Para processamento de imagens e correspondência de modelos.
Pytesseract: Para extração de texto via OCR.
Requests: Para envio de notificações via webhook (usado implicitamente em notify_webhooks).
Pydantic: Para validação e serialização de dados.
Bibliotecas Padrão: os, io, re, json, uuid, traceback, datetime, typing.

Requisitos adicionais:

Tesseract OCR: Deve estar instalado no sistema, com o caminho do executável configurado (C:\Program Files\Tesseract-OCR\tesseract.exe).
PostgreSQL: Um banco de dados PostgreSQL ativo com a URL de conexão especificada (postgresql://user:123456@localhost/nfse_db).


Configuração
A aplicação inclui as seguintes configurações:

Caminho do Tesseract:
pythonpytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
Especifica o caminho do executável do Tesseract OCR.
URL do Banco de Dados:
pythonDATABASE_URL = "postgresql://user:123456@localhost/nfse_db?client_encoding=utf8"
Configura a conexão com o banco de dados PostgreSQL.
Diretório Temporário:
pythonBASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
Cria um diretório temp no diretório de trabalho atual para armazenar PDFs temporariamente.
Configuração do SQLAlchemy:
pythonengine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Inicializa o motor do SQLAlchemy, a fábrica de sessões e a base declarativa para modelos ORM.


Modelos de Banco de Dados
A aplicação define dois modelos de banco de dados utilizando SQLAlchemy:
1. Task
Representa uma tarefa de processamento de um PDF de NFSe.
pythonclass Task(Base):
    __tablename__ = "task"
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="pendente")
    data_criacao = Column(DateTime, default=datetime.utcnow)
    data_conclusao = Column(DateTime, nullable=True)
    arquivo_pdf = Column(String, nullable=True)
    json_resultado = Column(Text, nullable=True)
    erro_mensagem = Column(String, nullable=True)

Campos:

id: Chave primária, inteiro auto-incrementado.
status: Status da tarefa (pendente, em andamento, concluída, falha).
data_criacao: Data e hora de criação da tarefa.
data_conclusao: Data e hora de conclusão da tarefa (opcional).
arquivo_pdf: Caminho do arquivo PDF carregado (opcional).
json_resultado: Resultado extraído em formato JSON (opcional).
erro_mensagem: Mensagem de erro em caso de falha (opcional).



2. Webhook
Armazena URLs de webhooks para notificações de eventos.
pythonclass Webhook(Base):
    __tablename__ = "webhook"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String)
    data_criacao = Column(DateTime, default=datetime.utcnow)
    actions = Column(String)

Campos:

id: Chave primária, inteiro auto-incrementado.
url: URL do webhook para receber notificações.
data_criacao: Data e hora de criação do webhook.
actions: Lista de ações separadas por vírgula para disparar notificações (e.g., upload, conclusao).



As tabelas são criadas no banco com:
pythonBase.metadata.create_all(bind=engine, checkfirst=True)

Esquemas Pydantic
Modelos Pydantic são usados para validação e serialização de requisições/respostas:
1. WebhookCreate
Define a estrutura para criação de um webhook.
pythonclass WebhookCreate(BaseModel):
    url: str
    actions: str

Campos:

url: URL do webhook (string).
actions: Ações separadas por vírgula às quais o webhook está inscrito (string).



2. WebhookResponse
Define a estrutura de resposta para dados de webhooks.
pythonclass WebhookResponse(BaseModel):
    id: int
    url: str
    data_criacao: datetime
    actions: str

Campos: Espelha o modelo de banco Webhook.

3. TaskResponse
Define a estrutura de resposta para dados de tarefas.
pythonclass TaskResponse(BaseModel):
    id: int
    status: str
    data_criacao: datetime
    data_conclusao: Optional[datetime] = None
    arquivo_pdf: Optional[str] = None
    erro_mensagem: Optional[str] = None

Campos: Espelha o modelo de banco Task, exceto por json_resultado.


Extrator de NFSe
A classe abstrata NFSeExtractor e sua implementação FortalezaNFSeExtractor lidam com a extração de dados de PDFs de NFSe.
1. NFSeExtractor (Classe Abstrata)
Define a interface para extratores de NFSe.
pythonclass NFSeExtractor(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> dict:
        pass

Método:

extract(file_path: str) -> dict: Método abstrato para extrair dados de um arquivo PDF.



2. FortalezaNFSeExtractor
Implementa a lógica de extração para o formato de NFSe de Fortaleza.
pythonclass FortalezaNFSeExtractor(NFSeExtractor):
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

Atributos:

template_path: Caminho para o logotipo de Fortaleza (brasao_fortaleza.png) usado na correspondência de modelo.
fixed_crops: Lista de coordenadas [y_start, y_end, x_start, x_end] para recortar campos específicos da imagem do PDF.



Métodos

pdf_to_image(pdf_path: str, page_num: int = 0) -> np.ndarray:
Converte a primeira página de um PDF em uma matriz NumPy (imagem) usando PyMuPDF com 300 DPI.

Entrada: Caminho do arquivo PDF, número da página (padrão: 0).
Saída: Imagem BGR como matriz NumPy.


extract_text_from_crop(img: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> str:
Extrai texto de uma região recortada da imagem usando Tesseract OCR.

Entrada: Matriz da imagem e coordenadas de recorte.
Saída: Texto extraído como string.
Processo:

Recorta a imagem usando as coordenadas fornecidas.
Converte o recorte para tons de cinza e aplica limiarização.
Usa Tesseract com idioma português e configurações específicas (--oem 3 --psm 6).




detect_brasao(img: np.ndarray) -> bool:
Verifica se o PDF é uma NFSe de Fortaleza detectando o logotipo da cidade por correspondência de modelo.

Entrada: Matriz da imagem.
Saída: Booleano indicando se o logotipo foi detectado (limiar: 0.6).
Processo:

Carrega o modelo do logotipo (brasao_fortaleza.png).
Aplica um filtro de nitidez ao modelo.
Realiza correspondência de modelo em múltiplas escalas com normalização do coeficiente de correlação.
Retorna True se uma correspondência ultrapassar o limiar.




parse_fortaleza(img: np.ndarray) -> dict:
Extrai dados estruturados de regiões de recorte predefinidas.

Entrada: Matriz da imagem.
Saída: Dicionário com campos extraídos:
python{
    "data_emissao": str,  # Formato ISO
    "numero_nfse": str,
    "prestador": {"nome": str, "cnpj": str, "endereco": str},
    "tomador": {"nome": str, "cpf_cnpj": str, "endereco": str},
    "servicos": [{"descricao": str, "quantidade": int, "valor_unitario": None, "valor_total": None}],
    "valores": {"valor_servicos": float, "valor_deducoes": float, "valor_iss": float, "valor_liquido": float}
}

Processo:

Extrai texto de cada região de recorte.
Usa expressões regulares para validar e formatar campos (e.g., data, CNPJ, valores numéricos).
Lida com dados ausentes ou inválidos de forma elegante.




extract(file_path: str) -> dict:
Método principal para processar um arquivo PDF.

Entrada: Caminho do arquivo PDF.
Saída: Dados extraídos ou mensagem de erro se o logotipo não for detectado.
Processo:

Converte o PDF em uma imagem.
Verifica a presença do logotipo de Fortaleza.
Se detectado, parseia os dados; caso contrário, retorna um erro.






Funções Auxiliares
safe_save_upload(upload: UploadFile, dest_folder: str) -> str
Salva um arquivo enviado em um caminho único no diretório especificado.

Entrada:

upload: Objeto UploadFile do FastAPI.
dest_folder: Diretório de destino (e.g., TEMP_DIR).


Saída: Caminho do arquivo salvo.
Processo:

Gera um nome de arquivo único usando carimbo de data/hora e UUID.
Salva o arquivo no diretório de destino.
Retorna o caminho completo do arquivo.




Notificação via Webhook
notify_webhooks(action: str, task_id: int)
Envia requisições HTTP POST para webhooks registrados para uma ação específica.

Entrada:

action: Tipo de evento (upload ou conclusao).
task_id: ID da tarefa que dispara a notificação.


Processo:

Consulta o banco de dados por webhooks inscritos na ação.
Envia uma requisição POST para cada URL de webhook com um payload JSON:
json{"action": action, "task_id": task_id, "timestamp": str}

Ignora falhas de webhooks individuais para garantir robustez.




Processamento em Segundo Plano
process_nfse(task_id: int)
Processa um PDF de NFSe em segundo plano.

Entrada: ID da tarefa.
Processo:

Recupera a tarefa do banco de dados.
Atualiza o status da tarefa para em andamento.
Extrai dados usando FortalezaNFSeExtractor.
Armazena o resultado como JSON em json_resultado ou uma mensagem de erro em erro_mensagem.
Atualiza o status da tarefa para concluída ou falha e define data_conclusao.
Notifica webhooks com a ação conclusao.
Exclui o arquivo PDF temporário.




Endpoints da API
A aplicação expõe os seguintes endpoints FastAPI:
1. GET /
Endpoint raiz para confirmar que a API está funcionando.

Resposta:
json{"message": "API de Leitura de NFSe rodando!"}


2. GET /health
Endpoint de verificação de saúde.

Resposta:
json{"status": "healthy"}


3. GET /upload-nfse
Fornece instruções para envio de arquivos NFSe.

Resposta:
json{"message": "Use POST /upload-nfse com multipart/form-data: file=@arquivo.pdf"}


4. POST /upload-nfse
Envia um PDF de NFSe para processamento.

Entrada: Dados de formulário multipart com um campo file (arquivo PDF).
Resposta: JSON com o ID da tarefa:
json{"task_id": int}

Processo:

Salva o PDF em TEMP_DIR com um nome único.
Cria uma Task com status pendente.
Enfileira process_nfse como tarefa em segundo plano.
Notifica webhooks com a ação upload.
Retorna o ID da tarefa.



5. GET /status/{task_id}
Recupera o status de uma tarefa.

Entrada: task_id (parâmetro de caminho).
Resposta:
json{
  "task_id": int,
  "status": str,
  "data_criacao": str,
  "data_conclusao": str|null
}

Erros:

404: Tarefa não encontrada.



6. GET /result/{task_id}
Recupera os dados extraídos de uma tarefa.

Entrada: task_id (parâmetro de caminho).
Resposta: JSON com os dados extraídos (de json_resultado).
Erros:

404: Tarefa não encontrada.
400: Nenhum resultado disponível.



7. POST /webhook
Registra um novo webhook.

Entrada: Corpo JSON (WebhookCreate):
json{
  "url": str,
  "actions": str
}

Resposta: JSON com detalhes do webhook (WebhookResponse).
Processo: Salva o webhook no banco de dados.

8. GET /webhook
Lista todos os webhooks registrados.

Resposta: Lista de objetos WebhookResponse.

9. GET /admin/tasks
Lista tarefas com filtragem opcional.

Parâmetros de Consulta:

limit: Número máximo de tarefas (1–1000, padrão: 100).
status: Filtro opcional por status (e.g., pendente, concluída).


Resposta: Lista de objetos TaskResponse.

10. GET /admin/webhooks
Lista webhooks.

Parâmetros de Consulta:

limit: Número máximo de webhooks (1–1000, padrão: 100).


Resposta: Lista de objetos WebhookResponse.


Uso

Configuração:

Instale as dependências: pip install fastapi uvicorn sqlalchemy psycopg2-binary pymupdf opencv-python pytesseract requests.
Instale o Tesseract OCR e configure seu caminho.
Configure um banco de dados PostgreSQL e atualize DATABASE_URL.
Coloque o arquivo brasao_fortaleza.png no diretório do projeto.


Executar a API:
bashuvicorn main:app --reload
A API estará disponível em http://localhost:8000.
Enviar uma NFSe:
bashcurl -X POST -F "file=@nfse.pdf" http://localhost:8000/upload-nfse
Resposta: {"task_id": 1}
Verificar Status:
bashcurl http://localhost:8000/status/1

Obter Resultado:
bashcurl http://localhost:8000/result/1

Registrar um Webhook:
bashcurl -X POST -H "Content-Type: application/json" -d '{"url":"https://example.com/webhook","actions":"upload,conclusao"}' http://localhost:8000/webhook

Listar Tarefas:
bashcurl http://localhost:8000/admin/tasks?limit=10&status=concluída



Tratamento de Erros

Erros de Upload: Retorna código 500 com detalhes do erro e rastreamento.
Tarefa Não Encontrada: Retorna código 404 para task_id inválido.
Resultado Não Disponível: Retorna código 400 se json_resultado estiver vazio.
Erros de OCR: Capturados em erro_mensagem e registrados na tarefa.
Falhas de Webhook: Ignoradas silenciosamente para evitar bloqueios.


Extensibilidade

Novos Formatos de NFSe:

Crie novas classes derivadas de NFSeExtractor para outros municípios.
Atualize as coordenadas de recorte e a lógica de parseamento conforme necessário.


Melhorias em Webhooks:

Adicione lógica de retentativa ou notificações baseadas em filas.
Suporte mais ações ou personalização de payloads.


Escalabilidade do Banco:

Adicione índices para campos frequentemente consultados.
Implemente paginação para listas grandes de tarefas.


Processamento de Imagens:

Melhore detect_brasao com correspondência de modelo mais robusta.
Adicione suporte para PDFs rotacionados ou desalinhados.
