import { ApiProperty } from '@nestjs/swagger'
import { IsNotEmpty, IsString } from 'class-validator'

export class CategorizeCsvDto {
    @ApiProperty({
        description: 'String do arquivo csv importado apos o parse',
        example: 'csv parseado',
    })
    @IsString()
    @IsNotEmpty()
    csvContent: string
}
