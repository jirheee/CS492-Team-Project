import axios from 'axios';

import config from '../../config';
import { RequestInterface } from '../../types/request';

const requestUrl = `http://${config.api.SERVER_URL}:${config.api.SERVER_PORT}/${config.api.PREFIX}`;

const Request: RequestInterface = {
  get: (path: string) => {
    console.log(`${requestUrl}${path}`);
    return axios.get(`${requestUrl}${path}`);
  },
  post: (path: string, data: Object) => axios.get(`${requestUrl}${path}`, data),
  delete: (path: string) => axios.get(`${requestUrl}${path}`)
};

export default Request;
