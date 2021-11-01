import { createConnection } from 'typeorm';
import config from '../config';

export default async () => {
  const { connectionOptions } = config;

  await createConnection(connectionOptions);

  // eslint-disable-next-line no-console
  console.log(`DB connected on ${connectionOptions.port}`);
};
