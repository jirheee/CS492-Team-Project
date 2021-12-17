import PythonSpawner from '../ml/pythonSpawner';

class TrainManager {
  private static t_manager: TrainManager;

  public trainingProcess = new Map<string, PythonSpawner>();

  public static getInstance(): TrainManager {
    if (!this.t_manager) {
      this.t_manager = new TrainManager();
    }
    return this.t_manager;
  }
}

export default TrainManager;
