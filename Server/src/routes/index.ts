import { Router } from 'express';

export default () => {
  const route = Router();
  route.get('/status', (req, res) => {
    console.log('status req');
    res.json({ data: 'Hello' });
  });
  return route;
};

// GET /api/statue
