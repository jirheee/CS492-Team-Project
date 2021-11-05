import express, { Express } from 'express';
import cors from 'cors';
import serverRoute from '../routes';

export default (app: Express) => {
  // TODO: Add Express middlewares

  const corsOptions = {
    origin: '*'
  };
  app.use(cors(corsOptions));
  app.use(express.json());

  app.use('/api', serverRoute());
};
