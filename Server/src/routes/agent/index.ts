import { Router } from 'express';
import { mkdir, writeFile } from 'fs';
import Agent from '../../entity/agent';
import { HyperParameters, Model } from '../../types/nn';

interface CreateModelInterface {
  name: string;
  model: Model;
}

interface TrainModelInterface {
  agentUUID: string;
  hyperparameters: HyperParameters;
}

export default (router: Router) => {
  const agentRouter = Router();
  router.use('/agent', agentRouter);

  agentRouter.post('/create', async (req, res) => {
    try {
      const { name, model }: CreateModelInterface = req.body;
      // TODO: See if model is valid

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
      const { agentUUID, hyperparameters }: TrainModelInterface = req.body;
      const agent = Agent.findOne(agentUUID);
    } catch (e) {
      console.error(e);
    }
  });
};
