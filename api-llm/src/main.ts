import { NestFactory } from '@nestjs/core'
import { AppModule } from './app.module'
import 'dotenv/config'
import { join } from 'path'
import { NestExpressApplication } from '@nestjs/platform-express'
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger'

async function bootstrap() {
    const app = await NestFactory.create<NestExpressApplication>(AppModule)

    app.useStaticAssets(join(__dirname, '..', 'public'))

    app.enableCors({
        origin: '*',
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
        allowedHeaders: ['Content-Type'],
    })

    const config = new DocumentBuilder()
        .setTitle('SiMCCIT - API de Categorização Terapêutica')
        .setDescription('API para categorização automática de falas terapêuticas.')
        .setVersion('1.0.0')
        .addTag('Categorização Terapêutica')
        .build()

    const documentFactory = () => SwaggerModule.createDocument(app, config)

    SwaggerModule.setup('docs', app, documentFactory)

    await app.listen(3000)
}
bootstrap()
