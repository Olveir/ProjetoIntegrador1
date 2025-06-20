import { HttpService } from '@nestjs/axios'
import { BadRequestException, InternalServerErrorException } from '@nestjs/common'
import { of } from 'rxjs'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CsvService } from './csv.service'
import { SimccitService } from './simccit.service'

const mockCsvService = {
    parseCsvToTranscription: vi.fn(),
    reconstructCsvWithCategorization: vi.fn(),
}

const mockHttpService = {
    post: vi.fn(),
}

const mockLogger = {
    debug: vi.fn(),
    error: vi.fn(),
}

const mockEnv = {
    API_URL: 'http://fake-api',
    API_KEY: 'fake-key',
    MODEL: 'fake-model',
}

vi.stubGlobal('process', { env: mockEnv })

describe('SimccitService', () => {
    let service: SimccitService

    beforeEach(() => {
        service = new SimccitService(mockHttpService as any as HttpService, mockCsvService as any as CsvService)
        // @ts-ignore
        service.logger = mockLogger
        vi.clearAllMocks()
    })

    afterEach(() => {
        vi.restoreAllMocks()
    })

    it('buildPrompt deve substituir {transcricao}', () => {
        const prompt = service['buildPrompt']('abc')
        expect(prompt).toContain('abc')
    })

    describe('callCategorizationApi', () => {
        it('deve lançar exceção se API_URL não existir', async () => {
            vi.stubGlobal('process', { env: { ...mockEnv, API_URL: undefined } })
            await expect(service['callCategorizationApi']('abc')).rejects.toThrow(InternalServerErrorException)
        })

        it('deve lançar exceção se MODEL não existir', async () => {
            vi.stubGlobal('process', { env: { ...mockEnv, MODEL: undefined } })
            await expect(service['callCategorizationApi']('abc')).rejects.toThrow(InternalServerErrorException)
        })

        it('deve chamar httpService.post com os parâmetros corretos', async () => {
            vi.stubGlobal('process', { env: mockEnv })
            const postMock = vi.fn().mockReturnValue(of({ data: { choices: [{ message: { content: '[]' } }] } }))
            mockHttpService.post = postMock
            await service['callCategorizationApi']('transcrição')
            expect(postMock).toHaveBeenCalledWith(
                mockEnv.API_URL,
                expect.objectContaining({
                    model: mockEnv.MODEL,
                    messages: expect.any(Array),
                    temperature: 0,
                }),
                expect.objectContaining({
                    headers: expect.objectContaining({
                        Authorization: expect.stringContaining('Bearer'),
                    }),
                }),
            )
        })
    })

    describe('categorizeTranscript', () => {
        it('deve retornar os dados parseados da API', async () => {
            const fakeResponse = {
                data: {
                    choices: [{ message: { content: '[{"falante":"Terapeuta","texto":"Oi","categoria":"SRE"}]' } }],
                },
            }
            vi.spyOn(service as any, 'callCategorizationApi').mockResolvedValue(fakeResponse)
            const result = await service.categorizeTranscript('Oi')
            expect(result).toEqual([{ falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' }])
        })

        it('deve lançar exceção e logar erro se a API falhar', async () => {
            vi.spyOn(service as any, 'callCategorizationApi').mockRejectedValue(new Error('fail'))
            await expect(service.categorizeTranscript('Oi')).rejects.toThrow()
            expect(mockLogger.error).toHaveBeenCalled()
        })
    })

    describe('categorizeCsv', () => {
        it('deve parsear, categorizar e reconstruir o CSV', async () => {
            mockCsvService.parseCsvToTranscription.mockReturnValue({ transcricao: 'Oi', records: [], headers: [] })
            vi.spyOn(service, 'categorizeTranscript').mockResolvedValue([
                { falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' },
            ])
            mockCsvService.reconstructCsvWithCategorization.mockReturnValue('csv-final')
            const result = await service.categorizeCsv('csv')
            expect(result).toBe('csv-final')
        })

        it('deve lançar InternalServerErrorException se não houver dados', async () => {
            mockCsvService.parseCsvToTranscription.mockReturnValue({ transcricao: 'Oi', records: [], headers: [] })
            vi.spyOn(service, 'categorizeTranscript').mockResolvedValue([])
            await expect(service.categorizeCsv('csv')).rejects.toThrow(InternalServerErrorException)
        })

        it('deve relançar exceções conhecidas', async () => {
            mockCsvService.parseCsvToTranscription.mockImplementation(() => {
                throw new BadRequestException('bad')
            })
            await expect(service.categorizeCsv('csv')).rejects.toThrow(BadRequestException)
        })

        it('deve encapsular erros desconhecidos', async () => {
            mockCsvService.parseCsvToTranscription.mockImplementation(() => {
                throw new Error('other')
            })
            await expect(service.categorizeCsv('csv')).rejects.toThrow(InternalServerErrorException)
        })
    })

    describe('processApiResponse', () => {
        it('deve lançar exceção se choices estiver vazio', () => {
            expect(() => service['processApiResponse']({ data: { choices: [] } })).toThrow(InternalServerErrorException)
        })

        it('deve parsear saída válida', () => {
            const response = {
                data: {
                    choices: [{ message: { content: '[{"falante":"Terapeuta","texto":"Oi","categoria":"SRE"}]' } }],
                },
            }
            const result = service['processApiResponse'](response)
            expect(result).toEqual([{ falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' }])
        })

        it('deve usar fallback se parseApiOutput lançar exceção', () => {
            const response = { data: { choices: [{ message: { content: 'invalid' } }] } }
            vi.spyOn(service as any, 'parseApiOutput').mockImplementation(() => {
                throw new Error('fail')
            })
            vi.spyOn(service as any, 'fallbackResponseParsing').mockReturnValue([
                { falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' },
            ])
            const result = service['processApiResponse'](response)
            expect(result).toEqual([{ falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' }])
        })
    })

    describe('validateJsonArrayStrict', () => {
        it('deve lançar exceção se não for array', () => {
            expect(() => service['validateJsonArrayStrict']('{}')).toThrow(BadRequestException)
        })
        it('deve lançar exceção se faltar chaves', () => {
            expect(() => service['validateJsonArrayStrict']('[{"falante":"a"}]')).toThrow(BadRequestException)
        })
        it('deve parsear array válido', () => {
            const arr = '[{"falante":"a","texto":"b","categoria":"c"}]'
            expect(service['validateJsonArrayStrict'](arr)).toEqual([{ falante: 'a', texto: 'b', categoria: 'c' }])
        })
    })

    describe('fallbackResponseParsing', () => {
        it('deve extrair dados com regex', () => {
            const output = '"falante":"Terapeuta","texto":"Oi","categoria":"SRE"'
            const result = service['fallbackResponseParsing'](output)
            expect(result).toEqual([{ falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' }])
        })
        it('deve lançar exceção se não encontrar nada', () => {
            expect(() => service['fallbackResponseParsing']('nada')).toThrow(InternalServerErrorException)
        })
    })
})
