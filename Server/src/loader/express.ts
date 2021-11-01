import { Express } from 'express';
import serverRoute from '../routes';

export default (app: Express) => {
  // TODO: Add Express middlewares

  app.use('/api', serverRoute);
};
