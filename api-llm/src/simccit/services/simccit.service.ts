import { HttpService } from '@nestjs/axios'
import { BadRequestException, Injectable, InternalServerErrorException, Logger } from '@nestjs/common'
import { lastValueFrom } from 'rxjs'
import { FalaCategorizada } from '../interfaces/FalaCategorizada.interface'
import { CsvService } from './csv.service'

@Injectable()
export class SimccitService {
    private readonly logger = new Logger(SimccitService.name)
    private SIMCCIT_PROMPT_TEMPLATE = `
            # Você é um especialista em categorização de interações terapêuticas pelo SiMCCIT.

            ## 1. Manual SiMCCIT (NÃO ALTERE)

            EIXO I-1 – Comportamento Verbal Vocal do Terapeuta

            - **SRE – Solicitação de Relato**  
            Solicita informações, sentimentos, pensamentos ou tendências à ação do cliente.  
            Exemplo: "Como você se sentiu quando ele te falou isso?"  
            Critério: Solicitar descrição de eventos, sentimentos ou pensamentos.

            - **FAC – Facilitação**  
            Verbalizações mínimas ou expressões que indicam atenção e sugerem continuidade do relato.  
            Exemplo: "Certo." / "Hum hum."  
            Critério: Só ocorre enquanto o cliente fala; não categorizar em pausas longas.

            - **EMP – Empatia**  
            Nomeação/inferência de sentimentos, validação, normalização, apoio, paráfrases, comentários de entendimento, humor, autorrevelações empáticas.  
            Exemplo: "Imagino o quanto isso te deixa ansioso."  
            Critério: Sugere acolhimento, aceitação, validação, sem julgamento.

            - **INF – Informação**  
            Fornecimento de dados, explicações, padrões, informações sobre terapia, estratégias, justificativas de intervenção.  
            Exemplo: "Biologia requer vários cursos adicionais de laboratório."  
            Critério: Relata fatos, explica eventos, descreve procedimentos.

            - **SRF – Solicitação de Reflexão**  
            Solicita ao cliente pensar, refletir, analisar, avaliar, prever ou observar eventos.  
            Exemplo: "O que você achou da reação dele?"  
            Critério: Solicita análise, avaliação, previsão ou observação.

            - **REC – Recomendação**  
            Sugestão de ações, conselhos, modelo, incentivo, estruturação de atividade, permissão/proibição.  
            Exemplo: "Tente conversar com seu pai durante a semana."  
            Critério: Especifica ações a serem executadas ou evitadas.

            - **INT – Interpretação**  
            Explicação de relações causais, padrões, diagnósticos, sínteses, metáforas, inferências, normalizações, previsões, discordâncias, indicações de contradição.  
            Exemplo: "Talvez o seu problema não seja de motivação, mas que até agora as coisas ainda não deram certo."  
            Critério: Acrescenta análise, síntese ou explicação nova ao relato do cliente.

            - **APR – Aprovação**  
            Elogios, avaliações positivas, relatos de ganhos, concordância, pseudodiscordância, sentimentos positivos, exclamações de aprovação, agradecimentos.  
            Exemplo: "Você tomou a decisão certa. Está lidando com isso muito bem!"  
            Critério: Julgamento favorável sobre ações ou características do cliente.

            - **REP – Reprovação**  
            Confronto, crítica, ironia, ameaça, paráfrase crítica, autorrevelação desafiadora, sentimentos negativos, advertência.  
            Exemplo: "As coisas são muito mais simples do que você pinta."  
            Critério: Julgamento desfavorável, crítica, ironia, ameaça.

            - **TOU – Outras Vocal Terapeuta**  
            Verbalizações não classificáveis nas categorias anteriores, comentários alheios ao tema, acertos ocasionais, recuperação de assunto, opiniões pessoais sobre eventos externos.  
            Exemplo: "Aceita uma balinha?"  
            Critério: Não se encaixa nas demais categorias.

            EIXO I-3 – Comportamento Verbal Vocal do Cliente

            - **SOL – Solicitação**  
            Cliente pede informações, avaliações, recomendações, procedimentos, asseguramento, apresenta demanda.  
            Exemplo: "Quanto tempo você acha que a terapia vai durar?"  
            Critério: Solicita algo ao terapeuta.

            - **REL – Relato**  
            Cliente descreve fatos, sentimentos, estados motivacionais, julgamentos, avaliações, relatos de registros.  
            Exemplo: "Eu fazia um acompanhamento de hepatite. Pedi para o médico fazer um teste de HIV."  
            Critério: Descreve eventos, sentimentos, opiniões.

            - **MEL – Melhora**  
            Relatos de ganhos terapêuticos, mudanças positivas, autocontrole, autoconsciência.  
            Exemplo: "Estou me sentindo muito bem. Depois da internação, todos os meus problemas acabaram."  
            Critério: Relata progresso, autocontrole, insight.

            - **MET – Metas**  
            Planejamento de estratégias, propostas de ações futuras.  
            Exemplo: "Vou ligar para ela e conversar sobre o que aconteceu."  
            Critério: Descreve planos, projetos, ações futuras.

            - **CER – Relações**  
            Estabelecimento de relações explicativas, identificação de padrões, atribuição de diagnóstico, síntese, inferências, previsões, reflexões.  
            Exemplo: "Eu acho que eu trabalho tanto porque assim eu evito discussões em casa."  
            Critério: Explica, relaciona, conclui, prevê.

            - **CON – Concordância**  
            Avaliações favoráveis sobre o terapeuta, relatos de esperança, satisfação, seguimento de recomendações, indicações de atenção, entendimento, exclamações de concordância.  
            Exemplo: "É. Você tem razão."  
            Critério: Expressa concordância, entendimento, satisfação.

            - **OPO – Oposição**  
            Queixas, descontentamento, contradição, sentimentos negativos, ironia, incredulidade, pedidos de interrupção, ameaças, recusas, desvios de assunto, recusa de elogios, relatos de não seguimento.  
            Exemplo: "Esta técnica de time-out não está funcionando com meu filho."  
            Critério: Discorda, recusa, critica, desafia.

            - **COU – Outras Vocal Cliente**  
            Acertos ocasionais, recuperação de assunto, respostas verbais não classificáveis, comentários alheios ao tema.  
            Exemplo: "Posso fumar?"  
            Critério: Não se encaixa nas demais categorias.

            ---

            ## 2. Categorias Permitidas

            - **Terapeuta (Eixo I-1):** SRE, FAC, EMP, INF, SRF, REC, INT, APR, REP, TOU  
            - **Cliente (Eixo I-3):** SOL, REL, MEL, MET, CER, CON, OPO, COU

            ---

            ## 3. Regras de Segmentação e Categorização

            - Analise cada verbalização completa de um falante por vez.
            - Divida a verbalização em segmentos menores sempre que identificar partes distintas que se enquadram em categorias diferentes, mesmo dentro da mesma frase.
            - Cada segmento deve ser o menor trecho de texto que se encaixa completamente em UMA ÚNICA categoria SiMCCIT do Eixo I-1 ou I-3, conforme as definições e critérios do manual.
            - **NUNCA** agrupe diferentes categorias em um mesmo segmento.
            - **NUNCA** inclua conteúdo fora das categorias listadas.
            - **SEMPRE** siga rigorosamente os critérios de inclusão/exclusão do manual.

            ---

            ## 4. Formato de Saída (OBRIGATÓRIO)

            - Responda **APENAS** com um array JSON.
            - Cada objeto deve conter:
            - "falante": string ("Terapeuta" ou "Cliente")
            - "texto": string (segmento exato)
            - "categoria": string (sigla SiMCCIT)
            - **NÃO** inclua explicações, introduções, conclusões ou qualquer texto fora do array JSON.

            ---

            ## 5. Exemplos CORRETOS

            [
                { "falante": "Terapeuta", "texto": "Como você está se sentindo hoje?", "categoria": "SRE" },
                { "falante": "Terapeuta", "texto": "Continue, estou ouvindo.", "categoria": "FAC" },
                { "falante": "Terapeuta", "texto": "Imagino que isso tenha sido difícil para você.", "categoria": "EMP" },
                { "falante": "Terapeuta", "texto": "Esse tipo de situação é comum em casos de ansiedade.", "categoria": "INF" },
                { "falante": "Terapeuta", "texto": "O que você acha que poderia fazer diferente da próxima vez?", "categoria": "SRF" },
                { "falante": "Terapeuta", "texto": "Sugiro que você tente anotar seus sentimentos ao longo do dia.", "categoria": "REC" },
                { "falante": "Terapeuta", "texto": "Talvez você esteja buscando aprovação dos outros por conta de experiências passadas.", "categoria": "INT" },
                { "falante": "Terapeuta", "texto": "Parabéns por ter conseguido se expressar tão claramente.", "categoria": "APR" },
                { "falante": "Terapeuta", "texto": "Não é adequado agir dessa forma.", "categoria": "REP" },
                { "falante": "Terapeuta", "texto": "Vamos retomar o que falamos na última sessão.", "categoria": "TOU" },
                { "falante": "Cliente", "texto": "Você pode me explicar por que isso acontece?", "categoria": "SOL" },
                { "falante": "Cliente", "texto": "Ontem eu fiquei muito ansioso no trabalho.", "categoria": "REL" },
                { "falante": "Cliente", "texto": "Tenho me sentido melhor desde a última sessão.", "categoria": "MEL" },
                { "falante": "Cliente", "texto": "Quero conseguir controlar minha ansiedade.", "categoria": "MET" },
                { "falante": "Cliente", "texto": "Acho que minha relação com meus pais influencia nisso.", "categoria": "CER" },
                { "falante": "Cliente", "texto": "Concordo com o que você disse.", "categoria": "CON" },
                { "falante": "Cliente", "texto": "Não acho que isso funcione para mim.", "categoria": "OPO" },
                { "falante": "Cliente", "texto": "Às vezes, só fico em silêncio e não sei o que dizer.", "categoria": "COU" }
            ]

            ---

            ## 6. Exemplos INCORRETOS (NÃO FAÇA)

            [{"falante": "Terapeuta", "texto": "Como você está? Imagino que seja difícil.", "categoria": "EMP"}]
            Motivo: Agrupa duas categorias diferentes em um único segmento.

            **NUNCA** devolva uma resposta que não seja um array JSON no formato acima.

            ---

            ## 7. Auto-checagem (OBRIGATÓRIO)

            Antes de finalizar, revise sua resposta e garanta que:
            - Só há um array JSON, sem texto extra.
            - Cada objeto tem as chaves: "falante", "texto", "categoria".
            - Não há exemplos anteriores, explicações ou qualquer outro texto fora do array.
            - Não há respostas de frases não existentes na Transcrição a ser categorizada.
            - Responda **apenas** com o array JSON, sem qualquer explicação, comentário, introdução, título ou texto antes ou depois do array.
            - **NÃO** escreva nada além do array JSON.
            - **NÃO** use blocos de código markdown.
            - **NÃO** escreva “Resposta:”, “Explicação:” ou qualquer outro texto fora do array.
            - Apenas o array JSON puro, começando com [ e terminando com ].

            ---

            ## 8. Transcrição a ser categorizada

            {transcricao}
            `
    constructor(
        private readonly httpService: HttpService,
        private readonly csvService: CsvService,
    ) {}

    private buildPrompt(transcricao: string): string {
        return this.SIMCCIT_PROMPT_TEMPLATE.replace('{transcricao}', transcricao)
    }

    async categorizeTranscript(transcricao: string): Promise<FalaCategorizada[]> {
        try {
            this.logger.debug('Iniciando a categorização dos dados')
            const apiResponse = await this.callCategorizationApi(transcricao)
            return this.processApiResponse(apiResponse)
        } catch (error) {
            this.logger.error(`Erro ao categorizar transcrição: ${error.message}`)
            throw error
        }
    }

    private async callCategorizationApi(transcricao: string) {
        const apiUrl = process.env.API_URL
        const apiKey = process.env.API_KEY
        const model = process.env.MODEL

        if (!apiUrl) throw new InternalServerErrorException('Configuração da API incompleta')
        if (!model) throw new InternalServerErrorException('Configuração do Modelo incompleta')

        const messages = [
            {
                role: 'user',
                content: this.buildPrompt(transcricao),
            },
        ]

        const headers = {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
        }

        const options = {
            temperature: 0,
            top_p: 0.1,
            top_k: 10,
            repeat_penalty: 1.2,
            repeat_last_n: 128,
            min_p: 0.05,
        }

        return lastValueFrom(
            this.httpService.post(
                apiUrl,
                {
                    model: model,
                    messages: messages,
                    options: options,
                },
                {
                    headers: headers,
                },
            ),
        )
    }

    async categorizeCsv(csvContent: string): Promise<string> {
        try {
            const parseResult = this.csvService.parseCsvToTranscription(csvContent)
            const categorizedData = await this.categorizeTranscript(parseResult.transcricao)

            if (!categorizedData?.length) {
                throw new InternalServerErrorException('Falha ao categorizar os dados')
            }
            this.logger.debug('Montando arquivo csv de retorno')
            return this.csvService.reconstructCsvWithCategorization(categorizedData)
        } catch (error) {
            this.logger.error(`Erro ao categorizar CSV: ${error.message}`)
            if (error instanceof BadRequestException || error instanceof InternalServerErrorException) {
                throw error
            }
            throw new InternalServerErrorException(error.message)
        }
    }

    private processApiResponse(response: any): FalaCategorizada[] {
        if (!response.data.choices?.length) {
            this.logger.error('Resposta da API inválida:', response.data)
            throw new InternalServerErrorException('Erro na resposta da API')
        }
        this.logger.debug('Resposta recebida e sendo tratada')

        const output = response.data.choices[0].message.content

        try {
            return this.parseApiOutput(output)
        } catch (error) {
            return this.fallbackResponseParsing(output)
        }
    }

    private parseApiOutput(output: string): FalaCategorizada[] {
        const jsonMatch = output.match(/^\s*\[\s*\{[\s\S]*\}\s*\]\s*$/m)
        let jsonStr = ''

        if (jsonMatch) {
            jsonStr = jsonMatch[0]
        } else {
            jsonStr = output.replace(/^\`\`\`json|^\`\`\`|\`\`\`$/gm, '').trim()

            const startIndex = jsonStr.indexOf('[')
            const endIndex = jsonStr.lastIndexOf(']')
            if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
                jsonStr = jsonStr.substring(startIndex, endIndex + 1)
            }
        }

        return this.validateJsonArrayStrict(jsonStr)
    }

    private validateJsonArrayStrict(output: string): FalaCategorizada[] {
        const trimmed = output.trim()
        if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) {
            throw new BadRequestException('A resposta não é um array JSON puro.')
        }
        let data: any
        try {
            data = JSON.parse(trimmed)
        } catch (e) {
            throw new BadRequestException('JSON inválido.')
        }
        if (!Array.isArray(data)) {
            throw new BadRequestException('A resposta não é um array JSON.')
        }
        for (const obj of data) {
            if (typeof obj !== 'object' || !('falante' in obj) || !('texto' in obj) || !('categoria' in obj)) {
                throw new BadRequestException('Um ou mais objetos não possuem as chaves obrigatórias.')
            }
        }
        return data as FalaCategorizada[]
    }

    private fallbackResponseParsing(output: string): FalaCategorizada[] {
        this.logger.debug('Tentando extrair categorias manualmente da resposta...')
        const regex = /"falante"\s*:\s*"([^"]+)"\s*,\s*"texto"\s*:\s*"([^"]+)"\s*,\s*"categoria"\s*:\s*"([^"]+)"/g
        const matches = [...output.matchAll(regex)]
        if (matches.length > 0) {
            const data: FalaCategorizada[] = matches.map((match) => ({
                falante: match[1],
                texto: match[2],
                categoria: match[3],
            }))
            this.logger.debug(`Extraídas ${data.length} categorizações manualmente`)
            return data
        }
        throw new InternalServerErrorException('Não foi possível extrair categorias da resposta')
    }
}
