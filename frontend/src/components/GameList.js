import React from 'react';

import { getWinnerPresentation } from '../lib/replayState';

function GameList({ games, loading, error, onRefresh, onSelectGame }) {
  if (loading) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-10 text-center">
        <div className="text-xl text-white">Loading games...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-rose-900/60 bg-rose-950/40 p-8 text-center">
        <p className="text-lg text-rose-200">{error}</p>
        <button
          type="button"
          onClick={onRefresh}
          className="mt-4 rounded-lg bg-rose-500 px-4 py-2 font-semibold text-white transition hover:bg-rose-400"
        >
          Retry
        </button>
      </div>
    );
  }

  if (games.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-8 text-center">
        <p className="text-lg text-slate-300">
          No games found. Run a game first to see replays here.
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6 shadow-xl shadow-slate-950/40">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Available Replays</h2>
          <p className="mt-1 text-sm text-slate-400">Choose a saved simulation to inspect agent behaviour and outcomes.</p>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-semibold text-slate-100 transition hover:border-sky-400 hover:text-sky-200"
        >
          Refresh
        </button>
      </div>
      <div className="grid gap-4">
        {games.map((game) => (
          <button
            type="button"
            key={game.replay_id || game.game_id}
            onClick={() => onSelectGame(game.replay_id || game.game_id)}
            className="rounded-xl border border-slate-800 bg-slate-800/80 p-5 text-left transition hover:-translate-y-0.5 hover:border-sky-500 hover:bg-slate-800"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-white">{game.game_id}</h3>
                <p className="mt-1 text-sm text-slate-400">
                  Condition: <span className="text-slate-200">{game.condition}</span> · Seed: <span className="text-slate-200">{game.seed}</span>
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Run: <span className="text-slate-300">{game.run_id || 'legacy'}</span>
                </p>
                <p className="mt-2 text-sm text-slate-500">
                  {game.config?.n_players || '?'} players · {game.config?.n_traitors || '?'} traitors · {game.rounds} rounds
                </p>
              </div>
              <div className="text-right">
                <div className={`text-lg font-bold ${getWinnerPresentation(game.winner).className}`}>
                  {getWinnerPresentation(game.winner).label}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

export default GameList;
