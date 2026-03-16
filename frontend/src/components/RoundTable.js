import React from 'react';
import AgentAvatar from './AgentAvatar';

function RoundTable({ gameState, minHeight = '600px' }) {
  if (!gameState?.players?.length) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-8 text-center">
        <p className="text-slate-400">No game data available</p>
      </div>
    );
  }

  const { currentEvent, players } = gameState;
  const numPlayers = players.length;

  const getPlayerPosition = (index) => {
    const angle = (index / numPlayers) * 2 * Math.PI - Math.PI / 2;
    const radius = 40;
    const x = 50 + radius * Math.cos(angle);
    const y = 50 + radius * Math.sin(angle);
    return { x, y };
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4" style={{ minHeight }}>
      <div className="relative w-full" style={{ aspectRatio: '1 / 1' }}>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="h-3/4 w-3/4 rounded-full border-8 border-amber-800 bg-gradient-to-br from-amber-900 to-amber-950 shadow-2xl">
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-4xl opacity-20">🎭</div>
            </div>
          </div>
        </div>

        {players.map((player, index) => {
          const pos = getPlayerPosition(index);

          return (
            <div
              key={player.id}
              className="absolute flex flex-col items-center"
              style={{
                left: `${pos.x}%`,
                top: `${pos.y}%`,
                transform: 'translate(-50%, -50%)'
              }}
            >
              <div className="relative">
                <AgentAvatar player={player} />
                <div className="absolute -bottom-6 left-1/2 whitespace-nowrap rounded bg-slate-800 px-2 py-1 text-xs font-bold text-white transform -translate-x-1/2">
                  P{player.id}
                </div>
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}

export default RoundTable;
