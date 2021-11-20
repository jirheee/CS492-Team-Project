import Model from './model';
import { HyperParameters } from './types';

class TrainInfo {
  public hyperparameters: HyperParameters;
  public nn_information: Model;

  constructor(hyperParameters: HyperParameters, model: Model) {
    this.hyperparameters = hyperParameters;
    this.nn_information = model;
  }
}

export default TrainInfo;
