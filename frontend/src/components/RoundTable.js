import React from 'react';
import AgentAvatar from './AgentAvatar';

function RoundTable({ gameState }) {
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
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-6" style={{ minHeight: '600px' }}>
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
              {player.speaking && player.message && (
                <div className="mb-2 max-w-xs">
                  <div className="relative rounded-lg bg-yellow-400 px-3 py-2 text-center text-sm font-semibold text-slate-950">
                    {player.message.substring(0, 80)}
                    {player.message.length > 80 ? '...' : ''}
                    <div className="absolute bottom-0 left-1/2 transform translate(-1/2, full) w-0 h-0 border-l-4 border-r-4 border-t-4 border-l-transparent border-r-transparent border-t-yellow-400"></div>
                  </div>
                </div>
              )}

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

      {currentEvent?.action_type === 'public_message' && (
        <div className="mt-6 rounded-xl bg-slate-800 p-4">
          <div className="flex items-start space-x-3">
            <div className="text-2xl">💬</div>
            <div>
              <div className="text-sm text-slate-400">
                Player {currentEvent.actor_id} says:
              </div>
              <div className="mt-1 text-white">
                {currentEvent.payload?.content}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RoundTable;
