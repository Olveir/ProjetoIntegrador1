import { join } from 'path'
import { Controller, Get, Res } from '@nestjs/common'
import { ApiOkResponse, ApiOperation } from '@nestjs/swagger'
import { Response } from 'express'

@Controller()
export class AppController {
    @Get('hello')
    @ApiOperation({
        summary: 'Rota para health da aplicação',
    })
    @ApiOkResponse({
        description: 'Hello World para teste health do sistema',
        type: 'string',
    })
    getHello(): string {
        return 'Hello World!'
    }

    @ApiOperation({
        summary: 'Rota para exposição da interface do projeto',
    })
    @ApiOkResponse({
        description: 'Interface gráfica do sistema',
    })
    @Get()
    root(@Res() res: Response) {
        res.sendFile(join(__dirname, '..', '..', 'public', 'index.html'))
    }
}
