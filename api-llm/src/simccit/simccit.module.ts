import { HttpModule } from '@nestjs/axios'
import { Module } from '@nestjs/common'
import { MulterModule } from '@nestjs/platform-express'
import { CsvService } from './services/csv.service'
import { SimccitService } from './services/simccit.service'
import { SimccitController } from './simccit.controller'

@Module({
    imports: [
        HttpModule,
        MulterModule.register({
            limits: {
                fileSize: 5 * 1024 * 1024,
            },
        }),
    ],
    controllers: [SimccitController],
    providers: [SimccitService, CsvService],
})
export class SimccitModule {}
