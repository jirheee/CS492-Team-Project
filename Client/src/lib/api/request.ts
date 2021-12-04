import axios from 'axios';

import config from '../../config';

const requestUrl = `http://${config.api.SERVER_URL}:${config.api.SERVER_PORT}/${config.api.PREFIX}`;

const customAxios = axios.create({ baseURL: 'http://localhost:5000/api' });

export default customAxios;
