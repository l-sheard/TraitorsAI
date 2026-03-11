import React, { useEffect, useState } from 'react';

import { fetchGames } from './api/replayApi';
import GameList from './components/GameList';
import GameViewer from './components/GameViewer';

function App() {
  const [games, setGames] = useState([]);
  const [selectedGameId, setSelectedGameId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadGames();
  }, []);

  const loadGames = async () => {
    try {
      setLoading(true);
      setError('');
      setGames(await fetchGames());
    } catch (error) {
      console.error('Error loading games:', error);
      setError('Unable to load replays. Check that the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <p className="mb-2 text-sm uppercase tracking-[0.3em] text-sky-300">Portfolio demo</p>
          <h1 className="text-4xl font-bold">Traitors AI Replay Console</h1>
          <p className="mt-3 max-w-3xl text-slate-300">
            A compact full-stack interface for browsing deterministic social-deduction simulations,
            inspecting event streams, and replaying agent decisions round by round.
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {!selectedGameId ? (
          <GameList
            games={games}
            loading={loading}
            error={error}
            onSelectGame={setSelectedGameId}
            onRefresh={loadGames}
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
