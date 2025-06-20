from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import pandas as pd
import logging
import numpy as np
from werkzeug.utils import secure_filename
from transformers import BertTokenizer, BertForSequenceClassification
import traceback
from io import StringIO

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=["*"])

# =============================================================================
# CONFIGURAÇÕES E VARIÁVEIS GLOBAIS
# =============================================================================

# Caminhos
MODEL_PATH = "models/llama-pth-checkpoint/melhor_modelo_pt.pt"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configurações do modelo (MESMAS do treinamento)
MAX_LENGTH = 256
NUM_CLASSES = 10

# Variáveis globais
model = None
tokenizer = None

# Mapeamento das categorias (MESMO do treinamento)
categorias_interacoes = {
    "APR": 0, "EMP": 1, "FAC": 2, "INF": 3, "INT": 4,
    "OVT": 5, "REC": 6, "SREL": 7, "SREF": 8, "REP": 9
}

# Mapeamento reverso
idx_to_categoria = {v: k for k, v in categorias_interacoes.items()}

# Nomes completos das categorias
nomes_categorias = {
    "APR": "Aprovação",
    "EMP": "Empatia", 
    "FAC": "Facilitação",
    "INF": "Informação",
    "INT": "Interpretação",
    "OVT": "Outras vocal terapeuta",
    "REC": "Recomendação",
    "SREL": "Solicitação de Relato",
    "SREF": "Solicitação de Reflexão", 
    "REP": "Reprovação"
}

# =============================================================================
# TRATAMENTO DE ERROS GLOBAL
# =============================================================================

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Erro não tratado: {e}")
    return jsonify({
        'error': 'Erro interno do servidor',
        'message': str(e),
        'type': 'internal_error'
    }), 500

# =============================================================================
# CARREGAMENTO DO MODELO E TOKENIZER
# =============================================================================

def load_model_and_tokenizer():
    """Carrega o modelo BERT e tokenizer"""
    global model, tokenizer
    
    try:
        logger.info("🔄 Carregando tokenizer BERT português...")
        
        # Carregar tokenizer (MESMO do treinamento)
        tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
        
        # Adicionar tokens especiais (MESMOS do treinamento)
        special_tokens = ["[CTX]", "[NOCTX]"]
        tokenizer.add_tokens(special_tokens)
        
        logger.info(f"✅ Tokenizer carregado. Vocabulário: {len(tokenizer)} tokens")
        
        logger.info(f"🔄 Carregando modelo de: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Arquivo do modelo não encontrado: {MODEL_PATH}")
            return False
        
        # Carregar modelo base BERT
        model = BertForSequenceClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", 
            num_labels=NUM_CLASSES
        )
        
        # Redimensionar embeddings para incluir tokens especiais
        model.resize_token_embeddings(len(tokenizer))
        
        # Carregar pesos treinados
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            # Se é state_dict
            model.load_state_dict(checkpoint, strict=False)
        else:
            # Se é o modelo completo
            model = checkpoint
        
        model.eval()
        
        # Teste rápido
        test_text = "Como você está se sentindo?"
        test_input = tokenizer(
            f"[NOCTX] Terapeuta: {test_text}",
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        with torch.no_grad():
            outputs = model(**test_input)
            logger.info(f"✅ Teste do modelo OK: input shape {test_input['input_ids'].shape} -> output shape {outputs.logits.shape}")
        
        logger.info("✅ Modelo e tokenizer carregados com sucesso!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# =============================================================================
# FUNÇÕES DE CLASSIFICAÇÃO
# =============================================================================

def preparar_texto(texto, contexto_cliente=""):
    """
    Prepara o texto no mesmo formato usado no treinamento
    """
    try:
        texto = str(texto).strip()
        contexto_cliente = str(contexto_cliente).strip()
        
        if contexto_cliente:
            # Com contexto do cliente
            texto_formatado = f"[CTX] Cliente: {contexto_cliente} Terapeuta: {texto}"
            tem_contexto = 1
        else:
            # Sem contexto do cliente  
            texto_formatado = f"[NOCTX] Terapeuta: {texto}"
            tem_contexto = 0
        
        logger.debug(f"Texto formatado: {texto_formatado[:100]}...")
        return texto_formatado, tem_contexto
        
    except Exception as e:
        logger.error(f"Erro ao preparar texto: {e}")
        return f"[NOCTX] Terapeuta: {texto}", 0

def classify_text(texto, contexto_cliente=""):
    """
    Classifica um texto usando o modelo BERT treinado
    """
    try:
        if model is None or tokenizer is None:
            return {
                'categoria': 'ERRO',
                'categoria_completa': 'Modelo não carregado',
                'confianca': 0.0,
                'erro': 'Modelo ou tokenizer não carregado'
            }
        
        # Preparar texto no formato correto
        texto_formatado, tem_contexto = preparar_texto(texto, contexto_cliente)
        
        # Tokenizar (MESMO processo do treinamento)
        inputs = tokenizer(
            texto_formatado,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        logger.debug(f"Input shape: {inputs['input_ids'].shape}")
        logger.debug(f"Primeiros tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:10])}")
        
        # Classificar
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Calcular probabilidades
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            logger.debug(f"Logits: {logits[0].tolist()}")
            logger.debug(f"Probabilities: {probabilities[0].tolist()}")
            logger.debug(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")
            
            # Mapear para categoria
            categoria_codigo = idx_to_categoria.get(predicted_class, f"CLASSE_{predicted_class}")
            categoria_nome = nomes_categorias.get(categoria_codigo, categoria_codigo)
            
            # Probabilidades por categoria
            prob_dict = {}
            for i in range(NUM_CLASSES):
                codigo = idx_to_categoria.get(i, f"CLASSE_{i}")
                nome = nomes_categorias.get(codigo, codigo)
                prob_dict[codigo] = {
                    'nome': nome,
                    'probabilidade': float(probabilities[0][i])
                }
            
            # Calcular entropia (medida de incerteza)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
            max_entropy = np.log(NUM_CLASSES)
            normalized_entropy = entropy / max_entropy
            
            resultado = {
                'categoria': categoria_codigo,
                'categoria_completa': categoria_nome,
                'confianca': float(confidence),
                'classe_id': predicted_class,
                'tem_contexto': tem_contexto,
                'texto_processado': texto_formatado,
                'probabilidades_todas': prob_dict,
                'metodo': 'bert_portuguese',
                'entropia': entropy,
                'entropia_normalizada': normalized_entropy,
                'tokens_utilizados': inputs['input_ids'].shape[1],
                'diagnostico': {
                    'max_length': MAX_LENGTH,
                    'logits_raw': logits[0].tolist(),
                    'num_tokens_especiais': len([t for t in tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) if t.startswith('[')]),
                    'formato_correto': texto_formatado.startswith('[CTX]') or texto_formatado.startswith('[NOCTX]')
                }
            }
            
            logger.info(f"Classificação: {categoria_codigo} ({categoria_nome}) - Confiança: {confidence:.3f}")
            return resultado
            
    except Exception as e:
        logger.error(f"Erro na classificação: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'categoria': 'ERRO',
            'categoria_completa': 'Erro na classificação',
            'confianca': 0.0,
            'erro': str(e)
        }

# =============================================================================
# ENDPOINTS DA API
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """Endpoint raiz"""
    return jsonify({
        'message': 'API Reptile - Sistema de Classificação BERT',
        'status': 'online',
        'version': '3.0_bert_fixed',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'categorias': list(nomes_categorias.keys()),
        'formato_texto': 'Use contexto_cliente (opcional) + texto_terapeuta'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Verificação de saúde"""
    return jsonify({
        'status': 'ok' if (model is not None and tokenizer is not None) else 'error',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'categorias_count': len(nomes_categorias),
        'model_path_exists': os.path.exists(MODEL_PATH),
        'max_length': MAX_LENGTH,
        'num_classes': NUM_CLASSES
    })

@app.route('/categorias', methods=['GET'])
def get_categories():
    """Retorna categorias disponíveis"""
    categorias_lista = []
    for codigo, nome in nomes_categorias.items():
        categorias_lista.append({
            'codigo': codigo,
            'nome': nome,
            'id': categorias_interacoes[codigo]
        })
    
    return jsonify({
        'categorias': categorias_lista,
        'total': len(categorias_lista),
        'mapeamento': categorias_interacoes
    })

@app.route('/classify', methods=['POST'])
def classify_single():
    """Classifica um texto"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON inválido'}), 400
        
        # Texto obrigatório
        texto = data.get('text', '').strip()
        if not texto:
            return jsonify({'error': 'Campo "text" é obrigatório'}), 400
        
        # Contexto opcional
        contexto_cliente = data.get('contexto_cliente', '').strip()
        
        resultado = classify_text(texto, contexto_cliente)
        
        return jsonify({
            'texto_terapeuta': texto,
            'contexto_cliente': contexto_cliente if contexto_cliente else None,
            'resultado': resultado
        })
        
    except Exception as e:
        logger.error(f"Erro no endpoint classify: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify-batch', methods=['POST'])
def classify_batch():
    """Classifica múltiplos textos"""
    try:
        data = request.get_json()
        
        # Formato 1: Lista de strings simples
        if 'texts' in data and isinstance(data['texts'], list):
            texts = data['texts']
            results = []
            
            for i, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    result = classify_text(text.strip())
                    results.append({
                        'indice': i,
                        'texto': text.strip(),
                        'resultado': result
                    })
        
        # Formato 2: Lista de objetos com texto e contexto
        elif 'interacoes' in data and isinstance(data['interacoes'], list):
            interacoes = data['interacoes']
            results = []
            
            for i, interacao in enumerate(interacoes):
                if isinstance(interacao, dict):
                    texto = interacao.get('texto', '').strip()
                    contexto = interacao.get('contexto_cliente', '').strip()
                    
                    if texto:
                        result = classify_text(texto, contexto)
                        results.append({
                            'indice': i,
                            'texto_terapeuta': texto,
                            'contexto_cliente': contexto if contexto else None,
                            'resultado': result
                        })
        else:
            return jsonify({
                'error': 'Formato inválido. Use "texts" para lista simples ou "interacoes" para lista com contexto'
            }), 400
        
        return jsonify({
            'resultados': results,
            'total_processados': len(results)
        })
        
    except Exception as e:
        logger.error(f"Erro no batch: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-tokenizer', methods=['POST'])
def test_tokenizer():
    """Testa o tokenizer com diferentes textos"""
    try:
        if tokenizer is None:
            return jsonify({'error': 'Tokenizer não carregado'}), 500
        
        data = request.get_json()
        textos_teste = data.get('textos', [
            "Como você está se sentindo hoje?",
            "Entendo que isso deve ser muito difícil para você",  
            "Isso é completamente normal e esperado",
            "Vamos focar no que podemos controlar agora",
            "Você mencionou que se sente muito ansioso"
        ])
        
        results = []
        
        for i, texto in enumerate(textos_teste):
            # Testar com e sem contexto
            for tem_contexto, contexto in [(False, ""), (True, "Estou me sentindo perdido ultimamente")]:
                texto_formatado, flag_contexto = preparar_texto(texto, contexto if tem_contexto else "")
                
                # Tokenizar
                tokens = tokenizer(
                    texto_formatado,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_LENGTH
                )
                
                # Converter para tokens legíveis
                token_ids = tokens['input_ids'][0].tolist()
                token_texts = tokenizer.convert_ids_to_tokens(token_ids)
                
                # Remover padding para visualização
                token_texts_clean = [t for t in token_texts if t != '[PAD]']
                
                results.append({
                    'indice': f"{i}_{['sem_ctx', 'com_ctx'][tem_contexto]}",
                    'texto_original': texto,
                    'contexto_cliente': contexto if tem_contexto else None,
                    'texto_formatado': texto_formatado,
                    'tem_contexto': flag_contexto,
                    'num_tokens': len(token_texts_clean),
                    'tokens_amostra': token_texts_clean[:15],  # Primeiros 15
                    'tokens_especiais': [t for t in token_texts_clean if t.startswith('[')],
                    'input_ids_shape': list(tokens['input_ids'].shape),
                    'attention_mask_sum': tokens['attention_mask'].sum().item()
                })
        
        return jsonify({
            'tokenizer_info': {
                'modelo': "neuralmind/bert-base-portuguese-cased",
                'vocab_size': len(tokenizer),
                'max_length': MAX_LENGTH,
                'tokens_especiais': ["[CTX]", "[NOCTX]", "[CLS]", "[SEP]", "[PAD]"]
            },
            'resultados': results
        })
        
    except Exception as e:
        logger.error(f"Erro no teste do tokenizer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/force-test-bert', methods=['POST'])
def force_test_bert():
    """Teste forçado com textos específicos para cada categoria"""
    try:
        if model is None or tokenizer is None:
            return jsonify({'error': 'Modelo ou tokenizer não carregado'}), 500
        
        # Textos específicos para cada categoria (baseados no domínio terapêutico)
        textos_teste = [
            ("Parabéns, você demonstrou muita coragem ao enfrentar isso", "APR"),  # Aprovação
            ("Entendo perfeitamente como isso deve ser difícil para você", "EMP"),  # Empatia
            ("Vamos continuar explorando essa questão importante", "FAC"),  # Facilitação
            ("Estudos mostram que essa abordagem tem eficácia comprovada", "INF"),  # Informação
            ("Isso pode significar que você está buscando mais segurança", "INT"),  # Interpretação
            ("Hmm, interessante essa sua colocação", "OVT"),  # Outras vocal terapeuta
            ("Reconheço que você fez um grande esforço", "REC"),  # Recomendação
            ("Conte-me mais sobre como foi essa experiência", "SREL"),  # Solicitação de Relato
            ("O que você pensa sobre essa situação?", "SREF"),  # Solicitação de Reflexão
            ("Você disse que se sente ansioso e preocupado", "REP")   # Reprovação
        ]
        
        results = []
        predictions = []
        
        for i, (texto, categoria_esperada) in enumerate(textos_teste):
            resultado = classify_text(texto)
            predictions.append(resultado['categoria'])
            
            results.append({
                'indice': i,
                'texto': texto,
                'categoria_esperada': categoria_esperada,
                'categoria_esperada_nome': nomes_categorias.get(categoria_esperada, categoria_esperada),
                'categoria_predita': resultado['categoria'],
                'categoria_predita_nome': resultado['categoria_completa'],
                'confianca': resultado['confianca'],
                'correto': resultado['categoria'] == categoria_esperada,
                'probabilidades_top3': sorted(
                    resultado['probabilidades_todas'].items(),
                    key=lambda x: x[1]['probabilidade'],
                    reverse=True
                )[:3]
            })
        
        # Análise dos resultados
        unique_predictions = len(set(predictions))
        total_predictions = len(predictions)
        acertos = sum(1 for r in results if r['correto'])
        
        analysis = {
            'total_textos': total_predictions,
            'predicoes_unicas': unique_predictions,
            'taxa_diferenciacao': unique_predictions / total_predictions,
            'sempre_mesma_classe': unique_predictions == 1,
            'distribuicao_classes': {cat: predictions.count(cat) for cat in set(predictions)},
            'acertos': acertos,
            'taxa_acerto': acertos / len(results),
            'classe_mais_predita': max(set(predictions), key=predictions.count) if predictions else None
        }
        
        # Recomendações
        recommendations = []
        if unique_predictions == 1:
            recommendations.append("🔴 CRÍTICO: Todas as predições são iguais")
            recommendations.append("💡 Verifique se o modelo foi carregado corretamente")
        elif analysis['taxa_acerto'] > 0.7:
            recommendations.append("✅ Modelo funcionando bem!")
        elif analysis['taxa_acerto'] > 0.3:
            recommendations.append("🟡 Modelo funcionando mas com precisão baixa")
        else:
            recommendations.append("🔴 Modelo com precisão muito baixa")
        
        return jsonify({
            'resultados': results,
            'analise': analysis,
            'recomendacoes': recommendations,
            'modelo_info': {
                'tokenizer': "neuralmind/bert-base-portuguese-cased",
                'max_length': MAX_LENGTH,
                'num_classes': NUM_CLASSES
            }
        })
        
    except Exception as e:
        logger.error(f"Erro no teste BERT: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/processar', methods=['POST'])
def processar_csv():
    """Processa arquivo CSV com estrutura: Fala_Cliente,Fala_Terapeuta"""
    try:
        # Verificação mais detalhada do modelo
        if model is None:
            logger.error("Modelo não carregado")
            return jsonify({'error': 'Modelo não carregado'}), 500
        
        if tokenizer is None:
            logger.error("Tokenizer não carregado")
            return jsonify({'error': 'Tokenizer não carregado'}), 500
        
        logger.info("Iniciando processamento de CSV...")
        
        # Verificar se arquivo foi enviado
        if 'file' not in request.files:
            logger.error("Nenhum arquivo enviado")
            return jsonify({'error': 'Arquivo não enviado'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            logger.error("Arquivo inválido")
            return jsonify({'error': 'Arquivo inválido'}), 400
        
        logger.info(f"Processando arquivo: {file.filename}")
        
        # Ler CSV com múltiplas tentativas de encoding
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                file.seek(0)  # Reset file pointer
                df = pd.read_csv(file, encoding=encoding)
                logger.info(f"CSV carregado com encoding: {encoding}")
                break
            except Exception as e:
                logger.warning(f"Falha ao ler com encoding {encoding}: {e}")
                continue
        
        if df is None:
            return jsonify({'error': 'Não foi possível ler o arquivo CSV'}), 400
        
        logger.info(f"CSV carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        logger.info(f"Colunas disponíveis: {list(df.columns)}")
        
        # Remover linhas completamente vazias
        df = df.dropna(how='all')
        logger.info(f"Após remover linhas vazias: {df.shape[0]} linhas")
        
        # Encontrar colunas (busca flexível)
        col_terapeuta = None
        col_cliente = None
        
        # Procurar coluna do terapeuta
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(termo in col_lower for termo in ['fala_terapeuta', 'terapeuta']):
                col_terapeuta = col
                logger.info(f"Coluna do terapeuta encontrada: {col}")
                break
        
        # Procurar coluna do cliente
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(termo in col_lower for termo in ['fala_cliente', 'cliente']):
                col_cliente = col
                logger.info(f"Coluna do cliente encontrada: {col}")
                break
        
        # Se não encontrou terapeuta, verificar se tem apenas 2 colunas
        if col_terapeuta is None:
            if len(df.columns) == 2:
                # Assumir que a segunda coluna é do terapeuta
                col_cliente = df.columns[0]
                col_terapeuta = df.columns[1]
                logger.info(f"Assumindo estrutura padrão: Cliente='{col_cliente}', Terapeuta='{col_terapeuta}'")
            elif len(df.columns) >= 1:
                col_terapeuta = df.columns[0]  # Primeira coluna como fallback
                logger.warning(f"Usando primeira coluna como terapeuta: {col_terapeuta}")
        
        if col_terapeuta is None:
            return jsonify({
                'error': f'Coluna do terapeuta não encontrada. '
                         f'Colunas disponíveis: {list(df.columns)}. '
                         f'Estrutura esperada: Fala_Cliente,Fala_Terapeuta'
            }), 400
        
        # Limpar e processar dados
        df[col_terapeuta] = df[col_terapeuta].fillna('').astype(str).str.strip()
        if col_cliente and col_cliente in df.columns:
            df[col_cliente] = df[col_cliente].fillna('').astype(str).str.strip()
        
        # Processar classificações
        results = []
        errors = []
        processed_count = 0
        
        logger.info(f"Processando {len(df)} linhas...")
        
        for idx, row in df.iterrows():
            try:
                texto_terapeuta = row[col_terapeuta].strip()
                
                # Pular linhas vazias
                if not texto_terapeuta or texto_terapeuta.lower() in ['nan', 'none', '']:
                    continue
                
                # Buscar contexto do cliente
                contexto_cliente = ""
                if col_cliente and col_cliente in df.columns:
                    contexto_cliente = row[col_cliente].strip()
                    if contexto_cliente.lower() in ['nan', 'none', '']:
                        contexto_cliente = ""
                
                # Classificar
                resultado = classify_text(texto_terapeuta, contexto_cliente)
                
                # Verificar se houve erro na classificação
                if 'erro' in resultado:
                    errors.append({
                        'linha': idx + 1,
                        'texto': texto_terapeuta[:100],
                        'erro': resultado['erro']
                    })
                else:
                    results.append({
                        'linha': idx + 1,
                        'fala_terapeuta': texto_terapeuta[:200] + '...' if len(texto_terapeuta) > 200 else texto_terapeuta,
                        'fala_cliente': contexto_cliente[:100] + '...' if len(contexto_cliente) > 100 else contexto_cliente if contexto_cliente else None,
                        'categoria_predita': resultado.get('categoria', 'ERRO'),
                        'categoria_nome': resultado.get('categoria_completa', 'Erro'),
                        'confianca': resultado.get('confianca', 0.0),
                        'tem_contexto': resultado.get('tem_contexto', 0)
                    })
                
                processed_count += 1
                
                # Log de progresso a cada 20 linhas
                if processed_count % 20 == 0:
                    logger.info(f"Processadas {processed_count} falas do terapeuta...")
                
            except Exception as e:
                logger.error(f"Erro ao processar linha {idx + 1}: {e}")
                errors.append({
                    'linha': idx + 1,
                    'texto': str(row.get(col_terapeuta, ''))[:100],
                    'erro': str(e)
                })
        
        logger.info(f"Processamento concluído: {len(results)} sucessos, {len(errors)} erros")
        
        # Calcular estatísticas apenas se há resultados
        estatisticas = {
            'total_linhas_csv': len(df),
            'linhas_processadas': len(results),
            'total_erros': len(errors),
            'estrutura_detectada': {
                'colunas': list(df.columns),
                'coluna_terapeuta': col_terapeuta,
                'coluna_cliente': col_cliente
            }
        }
        
        if results:
            # Estatísticas detalhadas
            categoria_counts = {}
            confianca_total = 0
            contexto_count = 0
            
            for result in results:
                cat = result['categoria_predita']
                categoria_counts[cat] = categoria_counts.get(cat, 0) + 1
                confianca_total += result['confianca']
                if result['tem_contexto']:
                    contexto_count += 1
            
            estatisticas.update({
                'categorias_preditas': categoria_counts,
                'confianca_media': confianca_total / len(results),
                'com_contexto_cliente': contexto_count,
                'sem_contexto_cliente': len(results) - contexto_count
            })
        
        response = {
            'sucesso': True,
            'resultados': results,
            'estatisticas': estatisticas
        }
        
        # Incluir erros se houver
        if errors:
            response['erros'] = errors[:10]  # Máximo 10 erros para não sobrecarregar
            response['total_erros'] = len(errors)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erro crítico no processamento CSV: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'sucesso': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/processar-csv', methods=['POST'])
def processar_csv_download():
    """Processa CSV com estrutura Fala_Cliente,Fala_Terapeuta e retorna arquivo CSV com classificações (SIMPLIFICADO)"""
    try:
        # Verificações iniciais
        if model is None or tokenizer is None:
            logger.error("Modelo ou tokenizer não carregado")
            return jsonify({'error': 'Modelo ou tokenizer não carregado'}), 500
        
        if 'file' not in request.files:
            logger.error("Arquivo não enviado")
            return jsonify({'error': 'Arquivo não enviado'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            logger.error("Arquivo inválido")
            return jsonify({'error': 'Arquivo inválido'}), 400
        
        logger.info(f"Processando CSV para download: {file.filename}")
        
        # Ler CSV
        df = None
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=encoding)
                logger.info(f"CSV carregado com encoding: {encoding}")
                break
            except Exception as e:
                logger.warning(f"Falha com encoding {encoding}: {e}")
                continue
        
        if df is None:
            logger.error("Não foi possível ler o arquivo CSV")
            return jsonify({'error': 'Não foi possível ler o arquivo CSV'}), 400
        
        # Remover linhas vazias
        df = df.dropna(how='all')
        logger.info(f"DataFrame carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Encontrar colunas (busca flexível)
        col_terapeuta = None
        col_cliente = None
        
        # Procurar coluna do terapeuta
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(termo in col_lower for termo in ['fala_terapeuta', 'terapeuta']):
                col_terapeuta = col
                logger.info(f"Coluna do terapeuta encontrada: {col}")
                break
        
        # Procurar coluna do cliente
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(termo in col_lower for termo in ['fala_cliente', 'cliente']):
                col_cliente = col
                logger.info(f"Coluna do cliente encontrada: {col}")
                break
        
        # Se não encontrou terapeuta, verificar se tem apenas 2 colunas
        if col_terapeuta is None:
            if len(df.columns) == 2:
                # Assumir que a segunda coluna é do terapeuta
                col_cliente = df.columns[0]
                col_terapeuta = df.columns[1]
                logger.info(f"Assumindo estrutura padrão: Cliente='{col_cliente}', Terapeuta='{col_terapeuta}'")
            else:
                logger.error(f"Não foi possível identificar colunas. Disponíveis: {list(df.columns)}")
                return jsonify({
                    'error': f'Não foi possível identificar colunas. '
                             f'Estrutura esperada: Fala_Cliente,Fala_Terapeuta. '
                             f'Encontrado: {list(df.columns)}'
                }), 400
        
        # Limpar dados
        df[col_terapeuta] = df[col_terapeuta].fillna('').astype(str).str.strip()
        if col_cliente and col_cliente in df.columns:
            df[col_cliente] = df[col_cliente].fillna('').astype(str).str.strip()
        
        # Criar lista para resultados SIMPLIFICADOS
        results_data = []
        processed_count = 0
        
        logger.info(f"Processando {len(df)} linhas...")
        
        for idx, row in df.iterrows():
            try:
                # Pegar dados originais
                fala_cliente = row[col_cliente] if col_cliente else ''
                fala_terapeuta = row[col_terapeuta]
                
                # Limpar dados
                fala_cliente = str(fala_cliente).strip() if fala_cliente else ''
                fala_terapeuta = str(fala_terapeuta).strip()
                
                # Processar fala do terapeuta
                if fala_terapeuta and fala_terapeuta.lower() not in ['nan', 'none', '']:
                    # Buscar contexto do cliente
                    contexto_cliente = ""
                    if fala_cliente and fala_cliente.lower() not in ['nan', 'none', '']:
                        contexto_cliente = fala_cliente
                    
                    # Classificar
                    resultado = classify_text(fala_terapeuta, contexto_cliente)
                    
                    # Adicionar resultado SIMPLIFICADO (apenas 3 colunas)
                    row_data = {
                        'Fala_cliente': fala_cliente if fala_cliente else '',
                        'Fala_terapeuta': fala_terapeuta,
                        'categoria': resultado.get('categoria', 'ERRO')
                    }
                    
                    processed_count += 1
                else:
                    # Linha vazia ou inválida
                    row_data = {
                        'Fala_cliente': fala_cliente if fala_cliente else '',
                        'Fala_terapeuta': fala_terapeuta,
                        'categoria': 'VAZIO'
                    }
                
                results_data.append(row_data)
                
                if processed_count % 20 == 0 and processed_count > 0:
                    logger.info(f"Processadas {processed_count} falas do terapeuta...")
                
            except Exception as e:
                logger.error(f"Erro na linha {idx + 1}: {e}")
                # Adicionar linha com erro
                row_data = {
                    'Fala_cliente': row.get(col_cliente, '') if col_cliente else '',
                    'Fala_terapeuta': row.get(col_terapeuta, ''),
                    'categoria': 'ERRO'
                }
                results_data.append(row_data)
        
        if not results_data:
            logger.error("Nenhuma linha válida foi processada")
            return jsonify({'error': 'Nenhuma linha válida foi processada'}), 400
        
        # Criar DataFrame de resultados com APENAS 3 COLUNAS
        df_results = pd.DataFrame(results_data)
        
        # Garantir ordem das colunas
        df_results = df_results[['Fala_cliente', 'Fala_terapeuta', 'categoria']]
        
        # Gerar CSV de saída
        output = StringIO()
        df_results.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        # Criar resposta
        csv_content = output.getvalue()
        output.close()
        
        # Nome do arquivo de saída
        original_name = file.filename.rsplit('.', 1)[0] if '.' in file.filename else file.filename
        output_filename = f"{original_name}_classificado.csv"
        
        # Preparar resposta HTTP
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename="{output_filename}"'
        response.headers['X-Total-Linhas'] = str(len(results_data))
        response.headers['X-Falas-Processadas'] = str(processed_count)
        response.headers['X-Formato'] = 'Fala_cliente,Fala_terapeuta,categoria'
        
        logger.info(f"CSV processado com sucesso: {len(results_data)} linhas, {processed_count} falas do terapeuta classificadas")
        logger.info(f"Formato de saída: 3 colunas (Fala_cliente, Fala_terapeuta, categoria)")
        
        return response
        
    except Exception as e:
        logger.error(f"Erro crítico no processamento CSV: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/debug-csv', methods=['POST'])
def debug_csv():
    """Debug do processamento CSV"""
    try:
        debug_info = {
            'model_loaded': model is not None,
            'tokenizer_loaded': tokenizer is not None,
            'file_received': 'file' in request.files
        }
        
        if 'file' in request.files:
            file = request.files['file']
            debug_info['filename'] = file.filename
            debug_info['file_valid'] = file and file.filename != ''
            
            if file and file.filename != '':
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='utf-8', nrows=5)
                    debug_info['csv_shape'] = list(df.shape)
                    debug_info['csv_columns'] = list(df.columns)
                    debug_info['sample_data'] = df.head(2).to_dict()
                except Exception as e:
                    debug_info['csv_error'] = str(e)
        
        return jsonify({
            'debug_info': debug_info,
            'model_type': str(type(model)) if model else None,
            'tokenizer_type': str(type(tokenizer)) if tokenizer else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# =============================================================================
# INICIALIZAÇÃO
# =============================================================================

logger.info("🚀 Iniciando API Reptile com BERT...")
model_loaded = load_model_and_tokenizer()

if model_loaded:
    logger.info("✅ Modelo BERT carregado com sucesso!")
else:
    logger.error("❌ Falha ao carregar modelo BERT!")

# Executar aplicação
if __name__ == '__main__':
    print(f"""
    🐍 API REPTILE COM BERT - v3.0_bert_fixed
    
    Status: {'✅ OK' if model_loaded else '❌ ERRO'}
    Modelo BERT: {'✅ Carregado' if model is not None else '❌ Não carregado'}
    Tokenizer: {'✅ Carregado' if tokenizer is not None else '❌ Não carregado'}
    Categorias: {len(nomes_categorias)} disponíveis
    
    🧠 CORREÇÕES IMPLEMENTADAS:
    - ✅ Tokenizer BERT português (neuralmind/bert-base-portuguese-cased)
    - ✅ Tokens especiais [CTX] e [NOCTX] adicionados
    - ✅ Formato de texto idêntico ao treinamento
    - ✅ Processamento com contexto cliente/terapeuta
    - ✅ Classificação com 10 categorias terapêuticas
    - ✅ Sistema de diagnóstico BERT
    - ✅ Endpoint /processar-csv para download
    - ✅ Nova estrutura: Fala_Cliente,Fala_Terapeuta
    
    📡 ENDPOINTS PRINCIPAIS:
    
    🔍 INFORMAÇÕES:
    - GET  /                     - Info da API
    - GET  /health               - Status completo
    - GET  /categorias           - Lista categorias
    
    🎯 CLASSIFICAÇÃO:
    - POST /classify             - Texto único (com contexto opcional)
    - POST /classify-batch       - Múltiplos textos
    
    🧪 TESTES BERT:
    - POST /test-tokenizer       - Testa tokenização
    - POST /force-test-bert      - Teste com categorias específicas
    
    📁 PROCESSAMENTO CSV:
    - POST /processar            - Retorna JSON com resultados
    - POST /processar-csv        - Baixa CSV classificado
    - POST /debug-csv            - Debug do processamento
    
    📋 FORMATO CSV:
    Fala_Cliente,Fala_Terapeuta
    "Estou ansioso","Como você está?"
    "","Parabéns pelo progresso"
    
    🎯 TESTE PRIORITÁRIO:
    Execute POST /force-test-bert para verificar funcionamento!
    
    {'🎉 Sistema BERT pronto!' if model_loaded else '⚠️  Verifique se o modelo BERT está correto'}
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)