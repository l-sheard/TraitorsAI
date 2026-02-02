import React from 'react';

function AgentAvatar({ player }) {
  const getRoleColor = () => {
    if (!player.alive) return 'bg-gray-600 border-gray-700';
    if (player.role === 'traitor') return 'bg-red-900 border-red-700';
    return 'bg-blue-900 border-blue-700';
  };

  const getStatusIcon = () => {
    if (!player.alive) return '💀';
    if (player.speaking) return '🗣️';
    if (player.role === 'traitor') return '🗡️';
    return '🛡️';
  };

  return (
    <div className="flex flex-col items-center">
      {/* Avatar circle */}
      <div
        className={`
          w-16 h-16 rounded-full border-4 flex items-center justify-center
          transition-all duration-300
          ${getRoleColor()}
          ${player.speaking ? 'ring-4 ring-yellow-400 scale-110' : ''}
          ${!player.alive ? 'opacity-50 grayscale' : ''}
        `}
      >
        <span className="text-2xl">{getStatusIcon()}</span>
      </div>

      {/* Player label */}
      <div className="mt-2 text-center">
        <div className={`
          text-sm font-semibold
          ${player.alive ? 'text-white' : 'text-gray-500'}
        `}>
          P{player.id}
        </div>
        {!player.alive && (
          <div className="text-xs text-red-500">Eliminated</div>
        )}
      </div>

      {/* Speech bubble for current speaker */}
      {player.speaking && player.message && (
        <div className="absolute -top-2 left-20 bg-gray-900 text-white text-xs rounded-lg p-2 shadow-lg max-w-xs z-10 border border-yellow-400">
          <div className="line-clamp-3">{player.message}</div>
          {/* Speech bubble arrow */}
          <div className="absolute left-0 top-1/2 transform -translate-x-2 -translate-y-1/2">
            <div className="w-0 h-0 border-t-4 border-b-4 border-r-4 border-transparent border-r-yellow-400"></div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AgentAvatar;
