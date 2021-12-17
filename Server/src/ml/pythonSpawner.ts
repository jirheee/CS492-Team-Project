import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ProcessType } from '../types';

interface TrainHistory {
  epoch: number;
  elapsed_time: number;
  loss: number;
  entropy: number;
  kl: number;
}

interface WinHistory {
  epoch: number;
  win_ratio: number;
}

class PythonSpawner {
  private process: ChildProcessWithoutNullStreams;

  private path: string;

  private file: string;

  private options: string[];

  public events: { onData: (data: string) => void; onExit?: () => void };

  private processType: ProcessType;

  private trainingHistory: TrainHistory[] = [];

  private winRateHistory: WinHistory[] = [];

  constructor(
    path: string,
    file: string,
    options: string[] = [],
    events: { onData: (data: string) => void; onExit?: () => void },
    processType: ProcessType = ProcessType.Train
  ) {
    this.path = path;
    this.file = file;
    this.options = options;
    this.events = events;
    this.processType = processType;
  }

  public run() {
    return new Promise((resolve, reject) => {
      const parameters = [this.file].concat(this.options);
      this.process = spawn('python3', parameters, { cwd: this.path });

      console.log('spawned process');
      console.log(this.process.spawnfile);

      this.process.on('disconnect', () => {
        console.log('disconected ');
      });

      this.process.on('exit', () => {
        console.log('exit');
        if (this.events.onExit !== undefined) {
          this.events.onExit();
        }
      });

      this.process.stdout.on('data', data => {
        try {
          const parsed: TrainHistory | WinHistory = JSON.parse(data);

          if (
            this.processType === ProcessType.Train &&
            Object.keys(parsed).find(key => key === 'loss')
          ) {
            this.trainingHistory.push(parsed as TrainHistory);
          }
          if (
            this.processType === ProcessType.Train &&
            Object.keys(parsed).find(key => key === 'win_ratio')
          ) {
            this.winRateHistory.push(parsed as WinHistory);
          }
          this.events.onData(String(data));
        } catch (e) {
          console.error('Parse failed', String(data));
        }
      });

      this.process.stdout.on('exit', () => {
        console.log('process finished');

        resolve('end');
      });

      this.process.stderr.on('data', data => {
        console.log(`error: ${data}`);
        reject(data);
      });
    });
  }

  public getTrainingHistory() {
    return this.trainingHistory;
  }

  public getWinRateHistory() {
    return this.winRateHistory;
  }
}

export default PythonSpawner;
