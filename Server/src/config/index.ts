import dotenv from 'dotenv';
import { ConnectionOptions } from 'typeorm';

const env = dotenv.config();
if (env.error) {
  throw new Error('No .env file found!');
}

const connectionOptions: ConnectionOptions = {
  type: 'mysql',
  host: process.env.DB_HOST,
  port: Number(process.env.DB_PORT),
  username: process.env.DB_USERNAME,
  password: process.env.DB_PASSWORD,
  database: process.env.MAINDB_NAME,
  synchronize: true,
  logging: false,
  entities: ['src/entity/**/*.ts']
};

export default {
  env: process.env.NODE_ENV ?? 'DEV',
  HTTP_PORT: Number(process.env.SERVER_PORT ?? 5000),
  connectionOptions
};
