📄 NFSe Extractor API

API em FastAPI para processamento de Notas Fiscais de Serviços Eletrônicas (NFS-e) em PDF.
A aplicação identifica o brasão da Prefeitura de Fortaleza e extrai informações estruturadas usando OCR (Tesseract + OpenCV + PyMuPDF).

⚙️ Funcionalidades

Upload de arquivos PDF contendo NFS-e.

Conversão do PDF para imagem com PyMuPDF.

Extração de texto por OCR com Tesseract.

Recorte de regiões fixas para capturar campos importantes (CNPJ, valores, serviços, etc).

Armazenamento do resultado em banco PostgreSQL.

Gerenciamento de tasks assíncronas com BackgroundTasks.

Sistema de webhooks para notificação de eventos:

upload: quando o PDF é enviado.

conclusao: quando o processamento termina (sucesso ou falha).

Endpoints administrativos para listar tarefas e webhooks.

📦 Requisitos

Python 3.9+

PostgreSQL

Tesseract OCR instalado no sistema

Exemplo de caminho no Windows:

C:\Program Files\Tesseract-OCR\tesseract.exe


Dependências Python:

pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic opencv-python-headless pytesseract PyMuPDF numpy requests

🔧 Configuração

Banco de Dados
Ajuste a URL no código para seu PostgreSQL:

DATABASE_URL = "postgresql://user:123456@localhost/nfse_db?client_encoding=utf8"


Tesseract OCR
Atualize o caminho se necessário:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


Diretórios temporários
Os arquivos enviados são salvos em ./temp durante o processamento e removidos ao final.

▶️ Como rodar
uvicorn main:app --reload


API rodará em:
👉 http://localhost:8000

🌐 Endpoints
🔹 Informativos

GET / → Mensagem inicial.

GET /health → Verifica se a API está saudável.

GET /upload-nfse → Instruções de uso do upload.

🔹 Upload e Processamento

POST /upload-nfse
Upload de uma NFS-e (arquivo PDF).
Parâmetros:

file: Arquivo PDF enviado via multipart/form-data.
Retorno:

{"task_id": <id>}

GET /status/{task_id}
Consulta o status de uma tarefa.
Retorno:

pendente, em andamento, concluída, ou falha.

GET /result/{task_id}
Obtém o resultado extraído em JSON.
Exemplo de retorno:

{
  "data_emissao": "2023-10-01T10:00:00",
  "numero_nfse": "12345",
  "prestador": {
    "nome": "Empresa Prestadora",
    "cnpj": "12.345.678/0001-90",
    "endereco": "Rua Exemplo, 123, Cidade, Estado"
  },
  "tomador": {
    "nome": "Cliente X",
    "cpf_cnpj": "98.765.432/0001-00",
    "endereco": "Av. Teste, 456, Cidade"
  },
  "servicos": [
    {
      "descricao": "Serviço de consultoria",
      "quantidade": 1,
      "valor_unitario": 1500.0,
      "valor_total": 1500.0
    }
  ],
  "valores": {
    "valor_servicos": 1500.0,
    "valor_deducoes": 0.0,
    "valor_iss": 45.0,
    "valor_liquido": 1455.0
  }
}

🔹 Webhooks

POST /webhook
Cadastra um novo webhook.
Body:

{
  "url": "https://meusistema.com/callback",
  "actions": "upload,conclusao"
}


GET /webhook
Lista todos os webhooks cadastrados.

🔹 Admin

GET /admin/tasks
Lista tarefas, com filtros de status e limit.

GET /admin/webhooks
Lista webhooks cadastrados.

🔄 Fluxo de Processamento

Usuário envia o PDF (/upload-nfse).

API cria uma Task com status "pendente".

BackgroundTasks inicia o processamento:

Converte PDF em imagem.

Detecta brasão da Prefeitura de Fortaleza.

Aplica OCR nas regiões definidas.

Estrutura os dados em JSON.

Atualiza o banco com:

Resultado JSON.

Status "concluída" ou "falha".

Dispara webhooks para os inscritos.

Remove arquivo temporário do servidor.

⚠️ Observações

Atualmente o extrator está otimizado para NFS-e de Fortaleza (com base no brasão e layout fixo).

Para outros municípios, será necessário implementar novas classes herdando de NFSeExtractor.

Arquivos enviados são removidos após o processamento.

Em caso de falha, o erro é salvo no campo erro_mensagem da task.
