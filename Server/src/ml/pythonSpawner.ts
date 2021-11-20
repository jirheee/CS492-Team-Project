import { ChildProcessWithoutNullStreams, spawn } from 'child_process';

class PythonSpawner {
  private process: ChildProcessWithoutNullStreams;

  private path: string;

  private options: string[];

  constructor(path: string, options: string[] = []) {
    this.path = path;
    this.options = options;
  }

  public run() {
    return new Promise((resolve, reject) => {
      this.process = spawn('python3', [this.path]);

      console.log('spawned process');
      console.log(this.process.spawnfile);

      this.process.on('disconnect', () => {
        console.log('disconected ');
      });

      this.process.stdout.on('data', data => {
        console.log(String(data));
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
