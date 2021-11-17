import { Router } from 'express';
import { mkdir, writeFile } from 'fs';
import Agent from '../../entity/agent';

export default (router: Router) => {
  const agentRouter = Router();
  router.use('/agent', agentRouter);

  agentRouter.post('/create', async (req, res) => {
    try {
      const { name, model } = req.body;
      const newAgent = new Agent();
      newAgent.name = name;
      const agent = await newAgent.save();
      const path = `src/ml/models/${agent.uuid}`;
      mkdir(path, () => {
        writeFile(`${path}/model.json`, JSON.stringify(model), () => {
          res.json({ model, status: 200 });
        });
      });
    } catch (e) {
      console.error(e);
    }
  });

  agentRouter.post('/train', async (req, res) => {
    try {
      const { agentUUID, hyperparameters } = req.body;
      const agent = Agent.findOne(agentUUID);
    } catch (e) {
      console.error(e);
    }
  });
};
