import { ChildProcessWithoutNullStreams, spawn } from 'child_process';
import { ProcessType } from '../types';

class PythonSpawner {
  private process: ChildProcessWithoutNullStreams;

  private path: string;

  private file: string;

  private options: string[];

  private events: { onData: (data: string) => void };

  private processType: ProcessType;

  constructor(
    path: string,
    file: string,
    options: string[] = [],
    events: { onData: (data: string) => void },
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

      this.process.stdout.on('data', data => {
        this.events.onData(String(data));
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
}

export default PythonSpawner;
