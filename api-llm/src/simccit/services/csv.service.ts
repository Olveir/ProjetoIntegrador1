import { BadRequestException, Injectable, Logger } from '@nestjs/common'
import { parse as csvParse } from 'csv-parse/sync'
import { stringify as csvStringify } from 'csv-stringify/sync'
import { FalaCategorizada } from '../interfaces/FalaCategorizada.interface'

@Injectable()
export class CsvService {
    private readonly logger = new Logger(CsvService.name)

    public parseCsvToTranscription(csvContent: string) {
        try {
            this.logger.debug(`Iniciando o parse do arquivo csv`)

            const cleanedCsv = this.cleanCsvContent(csvContent)
            const records = this.parseRecords(cleanedCsv)
            this.validateCsvStructure(records)

            const headers = Object.keys(records[0])
            const indices = this.getColumnIndices(headers)
            const transcricao = this.buildTranscricao(records)

            return {
                transcricao,
                headers,
                records,
                ...indices,
            }
        } catch (error) {
            this.logger.error(`Erro ao processar CSV: ${error.message}`)
            throw error
        }
    }

    private cleanCsvContent(csvContent: string): string {
        return csvContent
            .split('\n')
            .filter((line) => line.trim() && line.split(',').length >= 3)
            .join('\n')
    }

    private parseRecords(cleanedCsv: string) {
        return csvParse(cleanedCsv, {
            columns: true,
            skip_empty_lines: true,
            relax_column_count: true,
            trim: true,
        })
    }

    private validateCsvStructure(records: any[]) {
        if (records.length === 0) {
            throw new BadRequestException('Arquivo CSV vazio')
        }
        const headers = Object.keys(records[0])
        if (!headers.includes('falante') || !headers.includes('texto')) {
            throw new BadRequestException('O arquivo CSV não contém as colunas necessárias (falante, texto)')
        }
    }

    private getColumnIndices(headers: string[]) {
        return {
            falanteIndex: headers.indexOf('falante'),
            textoIndex: headers.indexOf('texto'),
            categorizacaoIndex: headers.indexOf('categoria'),
        }
    }

    private buildTranscricao(records: any[]): string {
        const transcricao = records
            .filter((record) => record.falante && record.texto)
            .map((record) => `${record.falante}: ${record.texto}`)
            .join('\n')

        if (!transcricao) {
            throw new BadRequestException('Nenhum dado válido encontrado no CSV')
        }
        return transcricao
    }

    public reconstructCsvWithCategorization(categorizedData: FalaCategorizada[]): string {
        try {
            const csv = csvStringify(categorizedData, {
                header: true,
                quoted: true,
                quoted_empty: true,
            })
            return csv
        } catch (error) {
            this.logger.error(`Erro ao reconstruir CSV: ${error.message}`)
            throw error
        }
    }
}
