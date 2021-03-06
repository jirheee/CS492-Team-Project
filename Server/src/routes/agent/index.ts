import { Router } from 'express';
import Agent from '../../entity/agent';
import {
  AgentUUID,
  HyperParameters,
  Model,
  TrainResponse,
  TrainStatus
} from '../../types/nn';
import {
  getAgentModel,
  saveModelJson,
  isValidHyperParameter,
  getAgentTrainInfo,
  saveTrainInfo,
  getAgentTrainHistory
} from './logic';

import PythonSpawner from '../../ml/pythonSpawner';
import { ProcessType } from '../../types';
import TrainManager from '../../manager/trainManager';

interface TrainModelInterface {
  agentUuid: AgentUUID;
  hyperparameters: HyperParameters;
}

export default (router: Router) => {
  const agentRouter = Router();
  router.use('/agent', agentRouter);

  agentRouter.post('/create', async (req, res) => {
    try {
      const model: Model = req.body;

      const newAgent = new Agent();
      newAgent.name = model.name;
      const agent = await newAgent.save();
      const modelPath = await saveModelJson(agent.uuid, model);

      res.json({
        model,
        name: model.name,
        modelPath,
        uuid: agent.uuid,
        status: 200
      });
    } catch (e) {
      console.error(e);
    }
  });

  agentRouter.post('/train', async (req, res) => {
    try {
      const { hyperparameters, agentUuid }: TrainModelInterface = req.body;
      if (!isValidHyperParameter(hyperparameters)) {
        throw new Error('Invalid HyperParameter');
      }
      const agent = await Agent.findOne(agentUuid);
      if (agent === undefined) {
        throw new Error('Invalid Agent UUID');
      }
      if (agent.trainStatus !== TrainStatus.NOT_TRAINED) {
        throw new Error(`TrainStatus ${agent.trainStatus}`);
      }

      agent.trainStatus = TrainStatus.TRAINING;
      await agent.save();

      await saveTrainInfo(hyperparameters, agentUuid);
      console.log(hyperparameters, agentUuid);

      const trainInfo: TrainResponse = {
        hyperparameters,
        trainStatus: agent.trainStatus
      };

      const model = await getAgentModel(agentUuid);
      // TODO: Start Training
      const process = new PythonSpawner(
        './src/ml/AlphaZero_Gomoku',
        'train.py',
        ['-u', agentUuid, '-r'],
        {
          onData: data => {
            console.log(data);
          },
          onExit: async () => {
            console.log('finished training');
            TrainManager.getInstance().trainingProcess.delete(agentUuid);
            agent.trainStatus = TrainStatus.TRAIN_FINISHED;
            await agent.save();
            console.log(agent);
          }
        },
        ProcessType.Train
      );

      TrainManager.getInstance().trainingProcess.set(agentUuid, process);

      process.run().catch(e => console.error(e));

      res.json({ model, trainInfo, status: 200 });
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

      const model = await getAgentModel(uuid);
      const hyperparameters = await getAgentTrainInfo(uuid);
      const trainInfo: TrainResponse = {
        hyperparameters,
        trainStatus: agent.trainStatus
      };
      const trainHistory = await getAgentTrainHistory(uuid);

      console.log(trainInfo);

      res.json({ model, trainInfo, status: 200, trainHistory, agent });
    } catch (e) {
      console.error(e);
    }
  });
};
