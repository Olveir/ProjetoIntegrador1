import { ApiProperty } from '@nestjs/swagger'
import { IsNotEmpty, IsString } from 'class-validator'

export class TranscricaoDto {
    @ApiProperty({
        description:
            'Transcrição das falas, uma por linha. Exemplo: "Terapeuta: Como você está?\\nCliente: Tenho me sentido ansioso."',
        example: 'Terapeuta: Como você está?\nCliente: Tenho me sentido ansioso.',
    })
    @IsString()
    @IsNotEmpty()
    transcricao: string
}
