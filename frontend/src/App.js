import React, { useState, useEffect } from 'react';
import axios from 'axios';
import GameList from './components/GameList';
import GameViewer from './components/GameViewer';

const API_BASE = 'http://localhost:8000';

function App() {
  const [games, setGames] = useState([]);
  const [selectedGameId, setSelectedGameId] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadGames();
  }, []);

  const loadGames = async () => {
    try {
      const response = await axios.get(`${API_BASE}/games`);
      setGames(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading games:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <header className="bg-gray-800 shadow-lg">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-white">
            🎭 Traitors AI - Game Viewer
          </h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {!selectedGameId ? (
          <GameList 
            games={games} 
            loading={loading} 
            onSelectGame={setSelectedGameId}
          />
        ) : (
          <GameViewer 
            gameId={selectedGameId} 
            onBack={() => setSelectedGameId(null)}
          />
        )}
      </main>
    </div>
  );
}

export default App;
