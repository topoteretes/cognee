import fs from 'fs';
import FormData from 'form-data';
import fetch from './fetch.js';

async function run() {
  try {
    // Default user is created automatically, you can create a new user if needed.
    // const registerResponse = await fetch('/v1/auth/register', {
    //   method: 'POST',
    //   body: {
    //     email: 'default_user@example.com',
    //     password: 'default_password',
    //     is_active: true,
    //     is_superuser: true,
    //     is_verified: true
    //   },
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    // });
    // const user = await registerResponse.json();

    const authCredentials = new FormData();
    authCredentials.append('username', 'default_user@example.com');
    authCredentials.append('password', 'default_password');
    
    const loginResponse = await fetch('/v1/auth/login', {
      method: 'POST',
      body: authCredentials,
    });

    const bearer = await loginResponse.json();
    const token = bearer.access_token;

    const response = await fetch('/v1/datasets', {}, token);
    const datasets = await response.json();
    console.log(datasets);

    const files = [
      fs.createReadStream('../data/artificial_intelligence.pdf'),
    ];

    const addData = new FormData();
    files.forEach((file) => {
      addData.append('data', file, file.name);
    })
    addData.append('datasetId', 'main');
    
    await fetch('/v1/add', {
      method: 'POST',
      body: addData,
      headers: addData.getHeaders(),
    }, token);

    await fetch('/v1/cognify', {
      method: 'POST',
      body: JSON.stringify({
        datasets: ['main'],
      }),
      headers: {
        'Content-Type': 'application/json',
      }
    }, token);

    const graphResponse = await fetch('/v1/datasets/main/graph', {
      method: 'GET',
    }, token);

    const graphUrl = await graphResponse.text();
    console.log('Graph URL:', graphUrl);

    // Search for summaries
    const summariesResponse = await fetch('/v1/search', {
      method: 'POST',
      body: JSON.stringify({
        searchType: 'SUMMARIES',
        query: 'Artificial Intelligence',
      }),
      headers: {
        'Content-Type': 'application/json',
      }
    }, token);


    const summariesResults = await summariesResponse.json();
    console.log('Summaries Results:', summariesResults);

    // Search for chunks
    const chunksResponse = await fetch('/v1/search', {
      method: 'POST',
      body: JSON.stringify({
        searchType: 'CHUNKS',
        query: 'Artificial Intelligence',
      }),
      headers: {
        'Content-Type': 'application/json',
      }
    }, token);

    const chunksResults = await chunksResponse.json();
    console.log('Chunks Results:', chunksResults);

    // Search for insights
    const insightsResponse = await fetch('/v1/search', {
      method: 'POST',
      body: JSON.stringify({
        searchType: 'INSIGHTS',
        query: 'Artificial Intelligence',
      }),
      headers: {
        'Content-Type': 'application/json',
      }
    }, token);

    const insightsResults = await insightsResponse.json();
    console.log('Insights Results:', insightsResults);
  } catch (error) {
    console.error('Error:', error);
  }
}

run();
