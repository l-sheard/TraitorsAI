import React from 'react';

function AgentAvatar({ player }) {
  const getRoleColor = () => {
    if (!player.alive) return 'bg-gray-600 border-gray-700';
    if (player.role === 'traitor') return 'bg-red-900 border-red-700';
    if (player.role === 'faithful') return 'bg-emerald-900 border-emerald-700';
    return 'bg-slate-700 border-slate-600';
  };

  const getStatusIcon = () => {
    if (!player.alive) return '💀';
    if (player.speaking) return '🗣️';
    if (player.role === 'traitor') return '🗡️';
    if (player.role === 'faithful') return '🛡️';
    return '🎭';
  };

  const getStatusLabel = () => {
    if (!player.alive) return player.eliminatedBy === 'murder' ? 'Murdered' : 'Banished';
    if (player.role === 'traitor') return 'Traitor';
    if (player.role === 'faithful') return 'Faithful';
    return 'Unknown role';
  };

  return (
    <div className="flex flex-col items-center">
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

      <div className="mt-2 text-center">
        <div className={`
          text-sm font-semibold
          ${player.alive ? 'text-white' : 'text-gray-500'}
        `}>
          P{player.id}
        </div>
        <div className="text-xs text-slate-400">{getStatusLabel()}</div>
      </div>
    </div>
  );
}

export default AgentAvatar;
