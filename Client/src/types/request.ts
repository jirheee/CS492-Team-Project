export interface RequestInterface {
  get: (path: string) => Promise<any>;
  post: (path: string, data: string) => Promise<any>;
  delete: (path: string) => Promise<any>;
}
