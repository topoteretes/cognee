import handleServerErrors from './handleServerErrors';

export default function fetch(url: string, options: RequestInit = {}): Promise<Response> {
  return global.fetch('http://127.0.0.1:8000/api' + url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
    },
  })
    .then(handleServerErrors);
}
