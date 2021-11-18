import { Router } from 'express';
import Agent from '../../entity/agent';
import { AgentUUID, HyperParameters, Model } from '../../types/nn';
import {
  getAgentModel,
  saveModelJson,
  isValidModel,
  isValidHyperParameter
} from './logic';

interface CreateModelInterface {
  name: string;
  model: Model;
}

interface TrainModelInterface {
  agentUUID: AgentUUID;
  hyperparameters: HyperParameters;
}

export default (router: Router) => {
  const agentRouter = Router();
  router.use('/agent', agentRouter);

  agentRouter.post('/create', async (req, res) => {
    try {
      const { name, model }: CreateModelInterface = req.body;
      if (!isValidModel(model)) {
        throw Error('Invalid Model Request');
      }

      const newAgent = new Agent();
      newAgent.name = name;
      const agent = await newAgent.save();
      const modelPath = saveModelJson(agent.uuid, model);

      res.json({ model, name, modelPath, status: 200 });
    } catch (e) {
      console.error(e);
    }
  });

  agentRouter.post('/train', async (req, res) => {
    try {
      const { agentUUID, hyperparameters }: TrainModelInterface = req.body;
      if (!isValidHyperParameter(hyperparameters)) {
        throw new Error('Invalid HyperParameter');
      }
      const agent = await Agent.findOne(agentUUID);
      if (agent === undefined) {
        throw new Error('Invalid Agent UUID');
      }

      const model = await getAgentModel(agentUUID);
      // TODO: Start Training

      res.json({ model, hyperparameters, agentUUID, statue: 200 });
    } catch (e) {
      console.error(e);
    }
  });

  agentRouter.get('/:uuid', async (req, res) => {
    try {
      const { uuid } = req.params;

      const agent = await Agent.findOne(uuid);
      if (agent === undefined) {
        throw new Error('Invalid Agent UUID');
      }

      throw new Error('not implemented');
    } catch (e) {
      console.error(e);
    }
  });
};
