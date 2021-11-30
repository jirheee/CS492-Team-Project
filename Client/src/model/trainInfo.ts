import { HyperParameters } from './types';

class TrainInfo {
  public hyperparameters: HyperParameters;
  public modelUuid: string;

  constructor(hyperParameters: HyperParameters, uuid: string) {
    this.hyperparameters = hyperParameters;
    this.modelUuid = uuid;
  }
}

export default TrainInfo;
