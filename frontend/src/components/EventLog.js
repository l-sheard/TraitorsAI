import React from 'react';

function EventLog({ events, currentIndex, onSelectEvent, maxHeight = '600px' }) {
  const getEventIcon = (event) => {
    switch (event.action_type) {
      case 'public_message': return '💬';
      case 'vote': return '🗳️';
      case 'banish_result': return '⚖️';
      case 'murder_result': return '🔪';
      case 'belief_update': return '🧠';
      case 'traitor_chat': return '🗡️';
      case 'assign_roles': return '🎲';
      case 'assign_persona': return '🎭';
      case 'game_end': return '🏁';
      default: return '📝';
    }
  };

  const getEventDescription = (event) => {
    switch (event.action_type) {
      case 'public_message':
        return `P${event.actor_id}: ${(event.payload?.content || '').substring(0, 50)}${(event.payload?.content || '').length > 50 ? '…' : ''}`;
      case 'vote':
        return `P${event.actor_id} votes for P${event.payload?.target_id}`;
      case 'banish_result':
        return event.payload?.eliminated ? `P${event.payload.eliminated} was banished` : 'Banishment unresolved';
      case 'murder_result':
        return event.payload?.eliminated ? `P${event.payload.eliminated} was murdered` : 'No murder occurred';
      case 'belief_update':
        return `P${event.actor_id} updated beliefs`;
      case 'traitor_chat':
        return `Traitor P${event.actor_id} strategizing`;
      case 'assign_roles':
        return 'Roles assigned';
      case 'assign_persona':
        return `P${event.actor_id} persona: ${event.payload?.persona?.name || 'unknown'}`;
      case 'game_end':
        return `Game ended: ${event.payload?.winner || 'unknown winner'}`;
      default:
        return event.action_type;
    }
  };

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4" style={{ maxHeight, overflowY: 'auto' }}>
      <h3 className="mb-4 text-xl font-bold text-white">Event Log</h3>
      <div className="space-y-2">
        {events.map((event, index) => (
          <div
            key={index}
            onClick={() => onSelectEvent(index)}
            className={`
              cursor-pointer rounded-lg p-3 transition-all
              ${index === currentIndex
                ? 'bg-sky-600 text-white shadow-lg shadow-sky-900/40'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
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
