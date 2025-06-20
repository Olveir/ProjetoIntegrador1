# SiMCCIT - Sistema de Categorização Terapêutica

<p align="center">
  <img src="https://nestjs.com/img/logo-small.svg" width="120" alt="Nest Logo" />
</p>

<p align="center">
  Sistema inteligente para categorização automática de falas terapêuticas utilizando IA
</p>

<p align="center">
  <a href="https://www.npmjs.com/~nestjscore" target="_blank">
    <img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" />
  </a>
  <a href="https://www.npmjs.com/~nestjcore" target="_blank">
    <img src="https://img.shields.io/npm/l/@nestjs/core.svg" alt="Package License" />
  </a>
</p>

---

## **Importante: Executando a API Llama (LLM)**

Para rodar a API Llama (LLM), você **deve primeiro navegar para a pasta `api-llm`**. Todas as instruções e passos a seguir nesta seção devem ser executados **dentro dessa pasta**.

---

## 📋 Índice

- [O que é o SiMCCIT?](#o-que-é-o-simccit)
- [Como Funciona?](#como-funciona)
- [Qual Opção Escolher?](#qual-opção-escolher)
- [Opção 1: Instalação Simples (Docker)](#opção-1-instalação-simples-docker)
- [Opção 2: Instalação Manual (Windows/Mac/Linux)](#opção-2-instalação-manual-windowsmaclinux)
- [Opção 3: Usando API Externa](#opção-3-usando-api-externa)
- [Como Usar o Sistema](#como-usar-o-sistema)
- [Exemplos Práticos](#exemplos-práticos)
- [Problemas Comuns e Soluções](#problemas-comuns-e-soluções)
- [Para Desenvolvedores](#para-desenvolvedores)

## 🎯 O que é o SiMCCIT?

O **SiMCCIT** é um sistema que ajuda profissionais da área de saúde mental a **categorizar automaticamente** as falas que acontecem durante sessões terapêuticas. 

### Em palavras simples:
- 📝 Você fornece o texto de uma conversa entre terapeuta e cliente
- 🤖 O sistema usa inteligência artificial para analisar cada fala
- 📊 Ele retorna cada fala com sua respectiva categoria (ex: "SREL", "REL", etc.)

### Por que usar?
- ⏰ **Economiza tempo**: Não precisa categorizar manualmente
- 🎯 **Mais preciso**: A IA analisa o contexto de cada fala
- 📈 **Facilita pesquisas**: Dados organizados para análise
- 💼 **Profissional**: Padroniza o processo de categorização

## 🔍 Como Funciona?

1. **Você envia** o texto da sessão terapêutica
2. **O sistema separa** cada fala por falante (Terapeuta/Cliente)
3. **A IA analisa** o conteúdo e contexto de cada fala
4. **Você recebe** o resultado com as categorias

**Exemplo:**
```
Entrada: "Terapeuta: Como você está? Cliente: Me sinto ansioso."
Saída: 
- Terapeuta: "Como você está?" → Categoria: "SREL"
- Cliente: "Me sinto ansioso." → Categoria: "REL"
```

## 🤔 Qual Opção Escolher?

### 🟢 **Opção 1: Docker (RECOMENDADA para iniciantes)**
- ✅ **Mais fácil**: Tudo configurado automaticamente
- ✅ **Funciona offline**: Não precisa de internet após instalação
- ✅ **Mais rápido**: Melhor performance
- ❌ **Requer**: Computador com pelo menos 8GB de RAM

### 🟡 **Opção 2: Instalação Manual**
- ✅ **Controle total**: Você configura cada parte
- ✅ **Flexível**: Pode usar diferentes modelos de IA
- ❌ **Mais complexo**: Requer mais passos
- ❌ **Mais tempo**: Instalação demorada

### 🔵 **Opção 3: API Externa**
- ✅ **Mais simples**: Não instala IA localmente
- ✅ **Menos recursos**: Funciona em computadores mais simples
- ❌ **Precisa de internet**: Sempre conectado
- ❌ **Pode ter custo**: Dependendo do serviço usado

---

## 🐳 Opção 1: Instalação Simples (Docker)

### **Passo 1: Instalar o Docker**

#### Windows:
1. Acesse: https://www.docker.com/products/docker-desktop/
2. Baixe o "Docker Desktop for Windows"
3. Execute o instalador e siga as instruções
4. Reinicie o computador quando solicitado

#### Mac:
1. Acesse: https://www.docker.com/products/docker-desktop/
2. Baixe o "Docker Desktop for Mac"
3. Arraste para a pasta Applications
4. Abra o Docker Desktop

#### Linux (Ubuntu/Debian):
```bash
# Atualize o sistema
sudo apt update

# Instale o Docker
sudo apt install docker.io docker-compose

# Adicione seu usuário ao grupo docker
sudo usermod -aG docker $USER

# Reinicie ou faça logout/login
```

### **Passo 2: Verificar se o Docker está funcionando**

Abra o **Terminal** (Windows: Prompt de Comando ou PowerShell) e digite:
```bash
docker --version
```

Se aparecer a versão, está funcionando! ✅

### **Passo 3: Baixar e executar o sistema**

1. **Baixe o projeto** (ou peça para alguém baixar para você)
2. **Abra o Terminal** na pasta do projeto
3. **Execute o comando:**

```bash
docker-compose up -d
```

### **Passo 4: Aguardar a instalação**

⏳ **Primeira vez**: Pode demorar 10-30 minutos (baixa o modelo de IA)
⏳ **Próximas vezes**: Inicia em 1-2 minutos

### **Passo 5: Verificar se está funcionando**

Abra seu navegador e acesse: **http://localhost:3000**

Se aparecer a tela do SiMCCIT, está pronto! 🎉

---

## 💻 Opção 2: Instalação Manual (Windows/Mac/Linux)

### **Passo 1: Instalar o Ollama**

#### Windows:
1. Acesse: https://ollama.ai/download/windows
2. Baixe o instalador
3. Execute e siga as instruções
4. Abra o **Prompt de Comando** e digite:
```bash
ollama pull llama3.1:8b
```
⏳ Aguarde o download (pode demorar bastante)

#### Mac:
1. Acesse: https://ollama.ai/download/mac
2. Baixe e instale
3. Abra o **Terminal** e digite:
```bash
ollama pull llama3.1:8b
```

#### Linux:
```bash
# Instale o Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Baixe o modelo
ollama pull llama3.1:8b
```

### **Passo 2: Instalar o Node.js**

1. Acesse: https://nodejs.org/
2. Baixe a versão **LTS** (recomendada)
3. Instale seguindo as instruções padrão

### **Passo 3: Configurar o projeto**

1. **Baixe o projeto** para seu computador
2. **Abra o Terminal** na pasta do projeto
3. **Instale as dependências:**
```bash
npm install
```

### **Passo 4: Iniciar os serviços**

1. **Em um Terminal**, inicie o Ollama:
```bash
ollama serve
```

2. **Em outro Terminal**, inicie o sistema:
```bash
npm run start
```

### **Passo 5: Acessar o sistema**

Abra seu navegador e acesse: **http://localhost:3000**

---

## 🌐 Opção 3: Usando API Externa

Se você quiser usar um serviço de IA online (como OpenAI, Anthropic, etc.) em vez de instalar localmente:

### **Passo 1: Configurar o arquivo .env**

1. **Abra o arquivo `.env`** na pasta do projeto
2. **Modifique as configurações:**

```env
# Para usar OpenAI (exemplo)
MODEL='gpt-3.5-turbo'
API_URL='https://api.openai.com/v1/chat/completions'
API_KEY='sua-chave-da-api-aqui'

# Para usar Anthropic (exemplo)
MODEL='claude-3-sonnet'
API_URL='https://api.anthropic.com/v1/messages'
API_KEY='sua-chave-da-api-aqui'

# Para usar Ollama local (padrão)
MODEL='llama3.1:8b'
API_URL='http://localhost:11434/v1/chat/completions'
```

### **Passo 2: Obter uma chave de API**

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Outros**: Consulte a documentação do serviço

### **Passo 3: Instalar apenas o sistema**

```bash
# Instale as dependências
npm install

# Inicie o sistema
npm run start
```

**💡 Dica**: Esta opção é mais simples, mas pode ter custos dependendo do serviço usado.

---

## 📖 Como Usar o Sistema

### **Método 1: Interface Web (Mais Fácil)**

1. **Abra seu navegador** e acesse: http://localhost:3000
2. **Escolha uma opção:**
   - **Texto**: Cole a transcrição diretamente
   - **Arquivo CSV**: Faça upload de um arquivo

#### **Para Texto:**
1. **Cole sua transcrição** no formato:
```
Terapeuta: Como você está se sentindo hoje?
Cliente: Tenho me sentido bastante ansioso ultimamente.
Terapeuta: Pode me contar mais sobre essa ansiedade?
```

2. **Clique em "Categorizar"**
3. **Aguarde o processamento** (alguns segundos)
4. **Veja os resultados** na tela

#### **Para Arquivo CSV:**
1. **Prepare seu arquivo** no formato:
```csv
falante,texto,categoria
Terapeuta,"Como você está se sentindo hoje?",
Cliente,"Tenho me sentido bastante ansioso ultimamente.",
```

2. **Clique em "Escolher arquivo"** ou arraste o arquivo
3. **Clique em "Categorizar"**
4. **Baixe o arquivo** com as categorias preenchidas

### **Método 2: Usando a API (Para Desenvolvedores)**

Se você tem conhecimento técnico, pode usar a API diretamente:

```bash
# Categorizar texto
curl -X POST http://localhost:3000/simccit/categorizar-texto \
  -H "Content-Type: application/json" \
  -d '{"transcricao": "Terapeuta: Como você está?\nCliente: Bem, obrigado."}'
```

---

## 💡 Exemplos Práticos

### **Exemplo 1: Sessão Terapêutica Básica**

**Entrada:**
```
Terapeuta: Bom dia! Como você está se sentindo hoje?
Cliente: Oi, doutor. Estou me sentindo um pouco ansioso.
Terapeuta: Entendo. Pode me contar o que está causando essa ansiedade?
Cliente: É o trabalho. Tenho muitas tarefas e pouco tempo.
Terapeuta: Que estratégias você tem usado para lidar com essa situação?
Cliente: Tentei fazer listas, mas não está funcionando muito bem.
```

**Resultado Esperado:**
- Terapeuta: "Bom dia! Como você está se sentindo hoje?" → **Pergunta sobre Estado**
- Cliente: "Estou me sentindo um pouco ansioso." → **Relato de Estado**
- Terapeuta: "Pode me contar o que está causando essa ansiedade?" → **Solicitação de Elaboração**
- Cliente: "É o trabalho. Tenho muitas tarefas e pouco tempo." → **Relato de Situação**
- Terapeuta: "Que estratégias você tem usado para lidar com essa situação?" → **Pergunta sobre Estratégias**
- Cliente: "Tentei fazer listas, mas não está funcionando muito bem." → **Relato de Tentativa**

### **Exemplo 2: Arquivo CSV**

**Arquivo de entrada (sessao.csv):**
```csv
falante,texto,categoria
Terapeuta,"Como foi sua semana?",
Cliente,"Foi difícil, tive alguns problemas no trabalho.",
Terapeuta,"Quer me contar sobre esses problemas?",
Cliente,"Meu chefe tem sido muito exigente ultimamente.",
```

**Arquivo de saída (sessao_categorizada.csv):**
```csv
falante,texto,categoria
Terapeuta,"Como foi sua semana?","Pergunta sobre Período"
Cliente,"Foi difícil, tive alguns problemas no trabalho.","Relato de Dificuldade"
Terapeuta,"Quer me contar sobre esses problemas?","Solicitação de Elaboração"
Cliente,"Meu chefe tem sido muito exigente ultimamente.","Relato de Situação"
```

---

## 🔧 Problemas Comuns e Soluções

### **❌ "O site não abre" (localhost:3000)**

**Possíveis causas:**
- O sistema não está rodando
- Outra aplicação está usando a porta 3000

**Soluções:**
1. **Verifique se está rodando:**
   ```bash
   # Para Docker
   docker-compose ps
   
   # Para instalação manual
   # Verifique se você executou "npm run start"
   ```

2. **Reinicie o sistema:**
   ```bash
   # Para Docker
   docker-compose restart
   
   # Para instalação manual
   # Pare com Ctrl+C e execute "npm run start" novamente
   ```

### **❌ "Erro de modelo não encontrado"**

**Causa:** O modelo de IA não foi baixado corretamente

**Solução:**
```bash
# Para Docker
docker-compose exec ollama ollama pull llama3.1:8b

# Para instalação manual
ollama pull llama3.1:8b
```

### **❌ "Sistema muito lento"**

**Causas possíveis:**
- Computador com pouca RAM
- Modelo de IA muito grande

**Soluções:**
1. **Use um modelo menor:**
   - Edite o arquivo `.env`
   - Mude `MODEL='llama3.1:8b'` para `MODEL='llama3.1:3b'`

2. **Feche outros programas** para liberar memória

3. **Use API externa** (Opção 3) se o computador for muito lento

### **❌ "Erro no arquivo CSV"**

**Causas possíveis:**
- Formato incorreto
- Caracteres especiais
- Codificação errada

**Soluções:**
1. **Use o formato correto:**
   ```csv
   falante,texto,categoria
   Terapeuta,"Sua fala aqui",
   Cliente,"Resposta aqui",
   ```

2. **Salve o arquivo em UTF-8** (no Excel: Salvar Como → Mais opções → UTF-8)

3. **Baixe o exemplo** clicando no botão na interface web

### **❌ "Docker não funciona"**

**Windows:**
- Certifique-se de que a virtualização está habilitada no BIOS
- Execute o Docker Desktop como administrador

**Mac:**
- Verifique se o Docker Desktop está rodando na barra de menu

**Linux:**
- Execute: `sudo systemctl start docker`

---

## 🛠️ Para Desenvolvedores

### **Estrutura do Projeto:**
```
src/
├── app.controller.ts          # Controller principal
├── app.module.ts             # Módulo principal
├── main.ts                   # Ponto de entrada
└── simccit/                  # Módulo de categorização
    ├── dto/                  # Data Transfer Objects
    ├── interfaces/           # Interfaces TypeScript
    ├── services/            # Lógica de negócio
    └── simccit.controller.ts # Controller da API
```

### **API Endpoints:**

#### **POST** `/simccit/categorizar-texto`
Categoriza uma transcrição de texto.

**Request:**
```json
{
  "transcricao": "Terapeuta: Como você está?\nCliente: Bem, obrigado."
}
```

**Response:**
```json
[
  {
    "falante": "Terapeuta",
    "texto": "Como você está?",
    "categoria": "Pergunta sobre Estado"
  },
  {
    "falante": "Cliente",
    "texto": "Bem, obrigado.",
    "categoria": "Resposta Positiva"
  }
]
```

#### **POST** `/simccit/categorizar-csv`
Processa arquivo CSV.

**Request:** Multipart form-data com arquivo CSV
**Response:** Arquivo CSV categorizado

#### **GET** `/simccit/example-csv-format`
Baixa exemplo de formato CSV.

### **Configuração Avançada:**

**Arquivo .env:**
```env
# Modelo de IA
MODEL='llama3.1:8b'

# URL da API
API_URL='http://localhost:11434/v1/chat/completions'

# Chave da API (se usando serviço externo)
API_KEY='sua-chave-aqui'

# Porta da aplicação
PORT=3000

# Ambiente
NODE_ENV=production
```

### **Scripts Disponíveis:**
```bash
npm run start          # Produção
npm run start:dev      # Desenvolvimento (auto-reload)
npm run start:debug    # Debug mode
npm run build          # Compilar
npm run test           # Testes
npm run format         # Formatar código
```

### **Docker Commands:**
```bash
# Iniciar serviços
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar serviços
docker-compose down

# Reconstruir
docker-compose build --no-cache
```

---

## 📞 Suporte e Ajuda

### **Documentação:**
- **API Swagger**: http://localhost:3000/docs (quando o sistema estiver rodando)
- **Ollama**: https://ollama.ai/docs
- **Docker**: https://docs.docker.com/

### **Precisa de Ajuda?**
1. **Verifique os problemas comuns** acima
2. **Consulte os logs** para ver mensagens de erro

### **Logs Úteis:**
```bash
# Docker
docker-compose logs -f nestjs
docker-compose logs -f ollama

# Instalação manual
# Os logs aparecem no terminal onde você executou npm run start
```