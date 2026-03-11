import axios from 'axios';

export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export async function fetchGames() {
  const { data } = await apiClient.get('/games');
  return data;
}

export async function fetchReplay(gameId) {
  const [summaryResponse, eventsResponse] = await Promise.all([
    apiClient.get(`/games/${gameId}/summary`),
    apiClient.get(`/games/${gameId}/events`),
  ]);

  return {
    summary: summaryResponse.data,
    events: eventsResponse.data,
  };
}
