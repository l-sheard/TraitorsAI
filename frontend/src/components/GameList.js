import React from 'react';

function GameList({ games, loading, onSelectGame }) {
  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-white text-xl">Loading games...</div>
      </div>
    );
  }

  if (games.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400 text-lg">
          No games found. Run a game first to see replays here.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg shadow-xl p-6">
      <h2 className="text-2xl font-bold text-white mb-6">Available Games</h2>
      <div className="grid gap-4">
        {games.map((game) => (
          <div
            key={game.game_id}
            onClick={() => onSelectGame(game.game_id)}
            className="bg-gray-700 hover:bg-gray-600 rounded-lg p-4 cursor-pointer transition-colors"
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-lg font-semibold text-white">
                  {game.game_id}
                </h3>
                <p className="text-gray-400 text-sm mt-1">
                  Condition: {game.condition} | Seed: {game.seed}
                </p>
              </div>
              <div className="text-right">
                <div className={`text-lg font-bold ${
                  game.winner === 'traitor' ? 'text-red-500' : 'text-blue-500'
                }`}>
                  {game.winner === 'traitor' ? '🗡️ Traitors Win' : '🛡️ Faithful Win'}
                </div>
                <p className="text-gray-400 text-sm mt-1">
                  {game.rounds} rounds
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GameList;
