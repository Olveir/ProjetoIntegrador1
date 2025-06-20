import { BadRequestException } from '@nestjs/common'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { CsvService } from './csv.service'

const csvValido = `falante,texto,categoria
Terapeuta,Oi,
Paciente,Olá,`

const csvInvalido = `nome,frase
A,B`

const csvVazio = ``

describe('CsvService', () => {
    let service: CsvService

    beforeEach(() => {
        service = new CsvService()
        vi.clearAllMocks()
    })

    it('parseCsvToTranscription deve retornar transcrição, headers e records', () => {
        const resultado = service.parseCsvToTranscription(csvValido)
        expect(resultado.transcricao).toContain('Terapeuta: Oi')
        expect(resultado.headers).toEqual(['falante', 'texto', 'categoria'])
        expect(resultado.records.length).toBe(2)
        expect(resultado.falanteIndex).toBe(0)
        expect(resultado.textoIndex).toBe(1)
        expect(resultado.categorizacaoIndex).toBe(2)
    })

    it('parseCsvToTranscription deve lançar exceção para CSV vazio', () => {
        expect(() => service.parseCsvToTranscription(csvVazio)).toThrow(BadRequestException)
    })

    it('parseCsvToTranscription deve lançar exceção para CSV sem colunas obrigatórias', () => {
        expect(() => service.parseCsvToTranscription(csvInvalido)).toThrow(BadRequestException)
    })

    it('parseCsvToTranscription deve lançar exceção se não houver dados válidos', () => {
        const csvSemDados = `falante,texto,categoria\n,,\n,,`
        expect(() => service.parseCsvToTranscription(csvSemDados)).toThrow(BadRequestException)
    })

    it('reconstructCsvWithCategorization deve atualizar e converter para CSV', () => {
        const parseResult = service.parseCsvToTranscription(csvValido)
        const categorizado = [
            { falante: 'Terapeuta', texto: 'Oi', categoria: 'SRE' },
            { falante: 'Paciente', texto: 'Olá', categoria: 'SA' },
        ]
        const csvFinal = service.reconstructCsvWithCategorization(categorizado)
        expect(csvFinal).toContain('"falante","texto","categoria"')
        expect(csvFinal).toContain('"Terapeuta","Oi","SRE"')
        expect(csvFinal).toContain('"Paciente","Olá","SA"')
    })
})
