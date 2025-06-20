import { Module } from '@nestjs/common'
import { AppController } from './app.controller'
import { SimccitModule } from './simccit/simccit.module'

@Module({
    imports: [SimccitModule],
    controllers: [AppController],
})
export class AppModule {}
