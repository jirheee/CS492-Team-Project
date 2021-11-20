import { Router } from 'express';
import agent from './agent';

export default () => {
  const route = Router();
  route.get('/status', (req, res) => {
    console.log('status req');
    res.json({ data: 'Hello' });
  });
  agent(route);
  return route;
};

// GET /api/statue
