import { readFile, writeFile, mkdir } from 'fs';
import { AgentUUID, Model } from '../../types/nn';

const getAgentDir = (uuid: AgentUUID) => `src/ml/models/${uuid}`;

const getAgentModelPath = (uuid: AgentUUID) =>
  `${getAgentDir(uuid)}/model.json`;

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
      writeFile(modelPath, JSON.stringify(model), () => {
        resolve(modelPath);
      });
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
  isValidHyperParameter
};
