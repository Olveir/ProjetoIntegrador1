import { ApiProperty } from '@nestjs/swagger'

export class UploadedFileDto {
    @ApiProperty({ description: 'Nome do campo no formulário', example: 'file' })
    fieldname: string

    @ApiProperty({
        description: 'Nome original do arquivo enviado',
        example: 'meuarquivo.csv',
    })
    originalname: string

    @ApiProperty({ description: 'Codificação do arquivo', example: '7bit' })
    encoding: string

    @ApiProperty({ description: 'Tipo MIME do arquivo', example: 'text/csv' })
    mimetype: string

    @ApiProperty({
        description: 'Conteúdo do arquivo em buffer',
        type: 'string',
        format: 'binary',
    })
    buffer: Buffer

    @ApiProperty({ description: 'Tamanho do arquivo em bytes', example: 12345 })
    size: number
}
