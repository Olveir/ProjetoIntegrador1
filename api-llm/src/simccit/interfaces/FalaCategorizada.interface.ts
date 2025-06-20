import { ApiProperty } from '@nestjs/swagger'

export class FalaCategorizada {
    @ApiProperty({
        example: 'Terapeuta',
        description: 'Identificação do falante',
    })
    falante: string

    @ApiProperty({ example: 'Como você está?', description: 'Texto da fala' })
    texto: string

    @ApiProperty({
        example: 'Pergunta aberta',
        description: 'Categoria atribuída à fala',
    })
    categoria: string
}
