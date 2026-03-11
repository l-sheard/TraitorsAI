function createInitialPlayers(summary) {
  const totalPlayers = summary?.config?.n_players || 0;
  const roleMap = summary?.roles || {};
  const players = {};

  for (let id = 1; id <= totalPlayers; id += 1) {
    players[id] = {
      id,
      role: roleMap[id] || roleMap[String(id)] || null,
      alive: true,
      speaking: false,
      message: '',
      eliminatedBy: null,
    };
  }

  return players;
}

export function buildReplaySnapshot(summary, events, currentEventIndex) {
  const players = createInitialPlayers(summary);
  const boundedIndex = Math.min(Math.max(currentEventIndex, 0), Math.max(events.length - 1, 0));
  const visibleEvents = events.slice(0, boundedIndex + 1);
  const currentEvent = visibleEvents[visibleEvents.length - 1] || null;

  visibleEvents.forEach((event) => {
    if (event.action_type === 'public_message' && players[event.actor_id]) {
      players[event.actor_id].message = event.payload?.content || '';
    }

    if (event.action_type === 'banish_result') {
      const eliminatedId = event.payload?.eliminated;
      if (players[eliminatedId]) {
        players[eliminatedId].alive = false;
        players[eliminatedId].eliminatedBy = 'banishment';
      }
    }

    if (event.action_type === 'murder_result') {
      const eliminatedId = event.payload?.eliminated;
      if (players[eliminatedId]) {
        players[eliminatedId].alive = false;
        players[eliminatedId].eliminatedBy = 'murder';
      }
    }
  });

  if (currentEvent?.action_type === 'public_message' && players[currentEvent.actor_id]) {
    players[currentEvent.actor_id].speaking = true;
  }

  return {
    currentEvent,
    events: visibleEvents,
    players: Object.values(players).sort((left, right) => left.id - right.id),
    round: currentEvent?.round || 0,
    phase: currentEvent?.phase || 'setup',
  };
}

export function getWinnerPresentation(winner) {
  if (winner === 'traitors') {
    return {
      label: '🗡️ Traitors win',
      className: 'text-red-400',
    };
  }

  if (winner === 'faithful') {
    return {
      label: '🛡️ Faithful win',
      className: 'text-emerald-400',
    };
  }

  return {
    label: '🤝 Draw',
    className: 'text-amber-300',
  };
}
