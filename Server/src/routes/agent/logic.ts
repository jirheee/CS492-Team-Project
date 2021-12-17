import { readFile, writeFile, mkdir } from 'fs';
import {
  AgentUUID,
  HyperParameters,
  Model,
  TrainHistory
} from '../../types/nn';

const getAgentDir = (uuid: AgentUUID) => `src/ml/models/${uuid}`;

const getAgentModelPath = (uuid: AgentUUID) =>
  `${getAgentDir(uuid)}/model.json`;

const getAgentTrainInfoPath = (uuid: AgentUUID) =>
  `${getAgentDir(uuid)}/train.json`;

const getAgentTrainHistoryPath = (uuid: AgentUUID) =>
  `${getAgentDir(uuid)}/output.json`;

const getAgentModel = (uuid: AgentUUID): Promise<Model> => {
  const modelPath = getAgentModelPath(uuid);
  return new Promise((resolve, reject) => {
    readFile(modelPath, 'utf-8', (err, data) => {
      if (err) reject(err);
      const model: Model = JSON.parse(data);
      resolve(model);
    });
  });
};

const saveModelJson = (uuid: AgentUUID, model: Model): Promise<string> => {
  return new Promise(resolve => {
    const agentDir = getAgentDir(uuid);
    const modelPath = getAgentModelPath(uuid);
    mkdir(agentDir, () => {
      writeFile(modelPath, JSON.stringify(model, null, 2), () => {
        resolve(modelPath);
      });
    });
  });
};

const getAgentTrainInfo = (uuid: string): Promise<HyperParameters> => {
  return new Promise(resolve => {
    const agentTrainInfoPath = getAgentTrainInfoPath(uuid);
    readFile(agentTrainInfoPath, 'utf-8', (err, data) => {
      if (err) {
        resolve({ lr: NaN, batch_size: NaN, buffer_size: NaN, epochs: NaN });
      } else {
        const hyperparameters: HyperParameters = JSON.parse(data);
        resolve(hyperparameters);
      }
    });
  });
};

const getAgentTrainHistory = (uuid: string): Promise<TrainHistory> => {
  return new Promise(resolve => {
    const agentTrainHistoryPath = getAgentTrainHistoryPath(uuid);
    readFile(agentTrainHistoryPath, 'utf-8', (err, data) => {
      if (err) {
        resolve({ start: '', train_progression: [], win_rates: [], end: '' });
      } else {
        const trainhistory: TrainHistory = JSON.parse(data);
        resolve(trainhistory);
      }
    });
  });
};

const saveTrainInfo = (
  hyperparameters: HyperParameters,
  uuid: string
): Promise<HyperParameters> => {
  return new Promise(resolve => {
    const trainInfoPath = getAgentTrainInfoPath(uuid);
    writeFile(trainInfoPath, JSON.stringify(hyperparameters, null, 2), () => {
      resolve(hyperparameters);
    });
  });
};

const isValidateBoard = (candidate: any): boolean => {
  if (candidate.board_width && candidate.board_height && candidate.n_in_row) {
    return true;
  }
  return false;
};

const isValidModel = (candidate: any) => {
  if (!candidate.board || !isValidateBoard(candidate.board)) {
    return false;
  }
  if (candidate.nn_type && candidate.layers && candidate.activ_func) {
    return true;
  }
  return false;
};

const isValidHyperParameter = (candidate: any) => {
  if (
    candidate.lr &&
    candidate.buffer_size &&
    candidate.batch_size &&
    candidate.epochs
  ) {
    return true;
  }
  return false;
};

export {
  getAgentDir,
  getAgentModelPath,
  getAgentModel,
  saveModelJson,
  isValidModel,
  isValidHyperParameter,
  getAgentTrainInfo,
  saveTrainInfo,
  getAgentTrainHistory
};
