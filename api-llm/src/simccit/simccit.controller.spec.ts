import { BadRequestException, StreamableFile } from '@nestjs/common'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SimccitController } from './simccit.controller'

const mockSimccitService = {
    categorizeTranscript: vi.fn(),
    categorizeCsv: vi.fn(),
}

describe('SimccitController', () => {
    let controller: SimccitController

    beforeEach(() => {
        controller = new SimccitController(mockSimccitService as any)
        vi.clearAllMocks()
    })

    it('categorizeTranscript deve lançar exceção se transcrição estiver vazia', async () => {
        await expect(controller.categorizeTranscript({ transcricao: '' })).rejects.toThrow(BadRequestException)
    })

    it('categorizeTranscript deve chamar o serviço e retornar resultado', async () => {
        const resultado = [{ falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' }]
        mockSimccitService.categorizeTranscript.mockResolvedValue(resultado)
        const retorno = await controller.categorizeTranscript({ transcricao: 'Oi' })
        expect(retorno).toEqual(resultado)
        expect(mockSimccitService.categorizeTranscript).toHaveBeenCalledWith('Oi')
    })

    it('categorizeCsvFile deve lançar exceção se arquivo não for enviado', async () => {
        await expect(controller.categorizeCsvFile(undefined as any)).rejects.toThrow(BadRequestException)
    })

    it('categorizeCsvFile deve lançar exceção se arquivo não for CSV', async () => {
        const file = { mimetype: 'image/png', originalname: 'foto.png', buffer: Buffer.from('') }
        await expect(controller.categorizeCsvFile(file as any)).rejects.toThrow(BadRequestException)
    })

    it('categorizeCsvFile deve chamar o serviço e retornar StreamableFile', async () => {
        const file = {
            mimetype: 'text/csv',
            originalname: 'teste.csv',
            buffer: Buffer.from('csv-content'),
        }
        mockSimccitService.categorizeCsv.mockResolvedValue('csv-categorizado')
        const retorno = await controller.categorizeCsvFile(file as any)
        expect(retorno).toBeInstanceOf(StreamableFile)
        const buffer = await retorno.getStream().read()
        expect(buffer.toString()).toContain('csv-categorizado')
        expect(mockSimccitService.categorizeCsv).toHaveBeenCalledWith('csv-content')
    })

    it('getExampleCsvFormat deve retornar um StreamableFile com o exemplo', () => {
        const retorno = controller.getExampleCsvFormat()
        expect(retorno).toBeInstanceOf(StreamableFile)
        const buffer = retorno.getStream().read()
        expect(buffer.toString()).toContain('falante,texto,categoria')
    })
})
