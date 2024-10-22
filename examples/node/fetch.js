import nodeFetch from 'node-fetch';
import handleServerErrors from './handleServerErrors.js';

export default function fetch(url, options = {}, token) {
  return nodeFetch('http://127.0.0.1:8000/api' + url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`,
    },
  })
    .then(handleServerErrors)
    .catch(handleServerErrors);
}
