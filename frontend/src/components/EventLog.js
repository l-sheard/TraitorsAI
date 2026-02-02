import React from 'react';

function EventLog({ events, currentIndex, onSelectEvent }) {
  const getEventIcon = (event) => {
    switch (event.action_type) {
      case 'public_message': return '💬';
      case 'vote': return '🗳️';
      case 'banish': return '⚖️';
      case 'murder': return '🔪';
      case 'belief_update': return '🧠';
      case 'traitor_chat': return '🗡️';
      default: return '📝';
    }
  };

  const getEventDescription = (event) => {
    switch (event.action_type) {
      case 'public_message':
        return `P${event.actor_id}: ${(event.payload?.content || '').substring(0, 50)}...`;
      case 'vote':
        return `P${event.actor_id} votes for P${event.payload?.target}`;
      case 'banish':
        return `P${event.payload?.player_id} was banished`;
      case 'murder':
        return `P${event.payload?.target} was murdered`;
      case 'belief_update':
        return `P${event.actor_id} updated beliefs`;
      case 'traitor_chat':
        return `Traitor P${event.actor_id} strategizing`;
      default:
        return event.action_type;
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4" style={{ maxHeight: '600px', overflowY: 'auto' }}>
      <h3 className="text-xl font-bold text-white mb-4">Event Log</h3>
      <div className="space-y-2">
        {events.map((event, index) => (
          <div
            key={index}
            onClick={() => onSelectEvent(index)}
            className={`
              p-3 rounded cursor-pointer transition-all
              ${index === currentIndex 
                ? 'bg-blue-600 text-white shadow-lg' 
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <div className="flex items-start space-x-2">
              <span className="text-lg">{getEventIcon(event)}</span>
              <div className="flex-1 min-w-0">
                <div className="text-xs opacity-75">
                  R{event.round} - {event.phase}
                </div>
                <div className="text-sm truncate">
                  {getEventDescription(event)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default EventLog;
