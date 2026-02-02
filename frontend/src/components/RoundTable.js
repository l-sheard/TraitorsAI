import React from 'react';
import AgentAvatar from './AgentAvatar';

function RoundTable({ gameState }) {
  if (!gameState?.events) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400">No game data available</p>
      </div>
    );
  }

  // Extract player info from events
  const getPlayerInfo = () => {
    const players = {};
    const { events, currentEvent } = gameState;
    
    // Find role assignments
    events.forEach(event => {
      if (event.action_type === 'assign_roles') {
        const roles = event.payload?.roles || {};
        Object.entries(roles).forEach(([playerId, role]) => {
          players[playerId] = {
            id: parseInt(playerId),
            role: role,
            alive: true,
            speaking: false
          };
        });
      }
    });

    // Track eliminations
    events.forEach(event => {
      if (event.action_type === 'banish' || event.action_type === 'murder') {
        const eliminated = event.payload?.player_id || event.payload?.target;
        if (eliminated && players[eliminated]) {
          players[eliminated].alive = false;
        }
      }
    });

    // Highlight current speaker
    if (currentEvent?.action_type === 'public_message') {
      const speakerId = currentEvent.actor_id;
      if (players[speakerId]) {
        players[speakerId].speaking = true;
        players[speakerId].message = currentEvent.payload?.content;
      }
    }

    return Object.values(players).sort((a, b) => a.id - b.id);
  };

  const players = getPlayerInfo();
  const numPlayers = players.length;

  // Calculate positions in a circle
  const getPlayerPosition = (index) => {
    const angle = (index / numPlayers) * 2 * Math.PI - Math.PI / 2; // Start at top
    const radius = 40; // percentage
    const x = 50 + radius * Math.cos(angle);
    const y = 50 + radius * Math.sin(angle);
    return { x, y };
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6" style={{ minHeight: '600px' }}>
      {/* Round Table */}
      <div className="relative w-full h-full" style={{ paddingBottom: '100%' }}>
        {/* Table surface */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-3/4 h-3/4 rounded-full bg-gradient-to-br from-amber-900 to-amber-950 border-8 border-amber-800 shadow-2xl">
            {/* Table center decoration */}
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-4xl opacity-20">🎭</div>
            </div>
          </div>
        </div>

        {/* Players positioned around the table */}
        {players.map((player, index) => {
          const pos = getPlayerPosition(index);
          return (
            <div
              key={player.id}
              className="absolute"
              style={{
                left: `${pos.x}%`,
                top: `${pos.y}%`,
                transform: 'translate(-50%, -50%)'
              }}
            >
              <AgentAvatar player={player} />
            </div>
          );
        })}
      </div>

      {/* Current message display */}
      {gameState.currentEvent?.action_type === 'public_message' && (
        <div className="mt-6 bg-gray-700 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <div className="text-2xl">💬</div>
            <div>
              <div className="text-sm text-gray-400">
                Player {gameState.currentEvent.actor_id} says:
              </div>
              <div className="text-white mt-1">
                {gameState.currentEvent.payload?.content}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RoundTable;
