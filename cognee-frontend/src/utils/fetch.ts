import handleServerErrors from './handleServerErrors';

export default function fetch(url: string, options: RequestInit = {}): Promise<Response> {
  return global.fetch('http://localhost:8000/api' + url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
    },
  })
    .then(handleServerErrors);
}
