import {
    BadRequestException,
    Body,
    Controller,
    Get,
    Post,
    StreamableFile,
    UploadedFile,
    UseInterceptors,
} from '@nestjs/common'
import { FileInterceptor } from '@nestjs/platform-express'
import {
    ApiBadRequestResponse,
    ApiBody,
    ApiConsumes,
    ApiExtraModels,
    ApiOkResponse,
    ApiOperation,
    ApiResponse,
    ApiTags,
    getSchemaPath,
} from '@nestjs/swagger'
import { TranscricaoDto } from './dto/Transficao.dto'
import { FalaCategorizada } from './interfaces/FalaCategorizada.interface'
import { UploadedFileDto } from './interfaces/UploadedFile.interface'
import { SimccitService } from './services/simccit.service'

@ApiTags('Categorização Terapêutica')
@ApiExtraModels(FalaCategorizada)
@Controller('simccit')
export class SimccitController {
    constructor(private readonly simccitService: SimccitService) {}

    @Post('categorizar-texto')
    @ApiOperation({
        summary: 'Categorização de transcrição de texto',
        description: 'Recebe uma transcrição de falas (texto livre) e retorna cada fala categorizada.',
    })
    @ApiBody({ type: TranscricaoDto })
    @ApiOkResponse({
        description: 'Lista de falas categorizadas',
        type: [FalaCategorizada],
        schema: {
            type: 'array',
            items: { $ref: getSchemaPath(FalaCategorizada) },
        },
        examples: {
            exemplo: {
                summary: 'Exemplo de categorização de falas',
                value: [
                    {
                        falante: 'Terapeuta',
                        texto: 'Como você está?',
                        categoria: 'SRE',
                    },
                    {
                        falante: 'Cliente',
                        texto: 'Tenho me sentido ansioso.',
                        categoria: 'REL',
                    },
                ],
            },
        },
    })
    @ApiBadRequestResponse({ description: 'Transcrição não pode estar vazia' })
    async categorizeTranscript(@Body() categorizeTranscriptDto: TranscricaoDto): Promise<FalaCategorizada[]> {
        if (!categorizeTranscriptDto.transcricao || categorizeTranscriptDto.transcricao.trim() === '') {
            throw new BadRequestException('Transcrição não pode estar vazia')
        }
        return this.simccitService.categorizeTranscript(categorizeTranscriptDto.transcricao)
    }

    @Post('categorizar-csv')
    @ApiOperation({
        summary: 'Categorização de arquivo CSV',
        description: 'Recebe um arquivo CSV com falas e retorna um novo CSV categorizado para download.',
    })
    @ApiConsumes('multipart/form-data')
    @ApiBody({
        schema: {
            type: 'object',
            properties: {
                file: {
                    type: 'string',
                    format: 'binary',
                    description: 'Arquivo CSV para categorização. Deve conter colunas "falante" e "texto".',
                },
            },
        },
    })
    @ApiOkResponse({
        description: 'CSV categorizado para download',
        schema: { type: 'string', format: 'binary' },
    })
    @ApiBadRequestResponse({ description: 'Arquivo não enviado ou inválido' })
    @UseInterceptors(FileInterceptor('file'))
    async categorizeCsvFile(@UploadedFile() file: UploadedFileDto) {
        if (!file) {
            throw new BadRequestException('Arquivo não enviado')
        }
        if (file.mimetype !== 'text/csv' && !file.originalname.endsWith('.csv')) {
            throw new BadRequestException('O arquivo deve ser um CSV')
        }
        const csvContent = file.buffer.toString('utf-8')
        const result = await this.simccitService.categorizeCsv(csvContent)
        return new StreamableFile(Buffer.from(result, 'utf-8'), {
            type: 'text/csv',
            disposition: `attachment; filename="categorized_${file.originalname}"`,
        })
    }

    @Get('example-csv-format')
    @ApiOperation({
        summary: 'Download de exemplo de formato CSV',
        description: 'Baixe um arquivo CSV de exemplo para saber o formato aceito pela API.',
    })
    @ApiOkResponse({
        description: 'Exemplo de CSV para download',
        schema: { type: 'string', format: 'binary' },
    })
    getExampleCsvFormat() {
        const exampleCsv = `falante,texto,categoria
  Terapeuta,"Bom dia, como você está se sentindo hoje?",
  Cliente,"Tenho me sentido bastante ansioso ultimamente, especialmente no trabalho."`

        return new StreamableFile(Buffer.from(exampleCsv, 'utf-8'), {
            type: 'text/csv',
            disposition: `attachment; filename="example_format.csv"`,
        })
    }
}
