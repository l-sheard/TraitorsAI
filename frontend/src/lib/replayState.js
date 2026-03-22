function createInitialPlayers(summary) {
  const totalPlayers = summary?.config?.n_players || 0;
  const roleMap = summary?.roles || {};
  const personaMap = summary?.personas || {};
  const aliasMap = summary?.player_aliases || {};
  const players = {};

  for (let id = 1; id <= totalPlayers; id += 1) {
    const persona = personaMap[id] || personaMap[String(id)] || {};
    const personaName = typeof persona?.name === 'string' ? persona.name : `Player ${id}`;
    const alias = aliasMap[id] || aliasMap[String(id)] || null;
    const displayName = alias ? `${alias} · ${personaName}` : personaName;
    players[id] = {
      id,
      role: roleMap[id] || roleMap[String(id)] || null,
      alias,
      personaName,
      displayName,
      alive: true,
      speaking: false,
      message: '',
      eliminatedBy: null,
    };
  }

  return players;
}

function buildVoteSummary(visibleEvents, currentRound) {
  if (!currentRound) {
    return {
      banishVotes: [],
      revote: [],
      murderVotes: [],
      banishCounts: {},
      revoteCounts: {},
      murderCounts: {},
    };
  }

  const banishVoteMap = new Map();
  const revoteMap = new Map();
  const murderVoteMap = new Map();

  visibleEvents.forEach((event) => {
    if (event.round !== currentRound) {
      return;
    }

    if (event.action_type === 'vote' && event.phase === 'voting') {
      banishVoteMap.set(event.actor_id, event.payload?.target_id ?? null);
    }

    if (event.action_type === 'vote' && event.phase === 'revote') {
      revoteMap.set(event.actor_id, event.payload?.target_id ?? null);
    }

    if (event.action_type === 'murder' && event.phase === 'murder') {
      murderVoteMap.set(event.actor_id, event.payload?.target_id ?? null);
    }
  });

  const toRows = (voteMap) => Array.from(voteMap.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([voterId, targetId]) => ({ voterId, targetId }));

  const toCounts = (rows) => rows.reduce((accumulator, row) => {
    if (row.targetId == null) {
      return accumulator;
    }
    const key = String(row.targetId);
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, {});

  const banishVotes = toRows(banishVoteMap);
  const revote = toRows(revoteMap);
  const murderVotes = toRows(murderVoteMap);

  return {
    banishVotes,
    revote,
    murderVotes,
    banishCounts: toCounts(banishVotes),
    revoteCounts: toCounts(revote),
    murderCounts: toCounts(murderVotes),
  };
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
    voteSummary: buildVoteSummary(visibleEvents, currentEvent?.round || 0),
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
