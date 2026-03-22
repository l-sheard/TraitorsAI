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

  const { currentEvent, players, voteSummary } = gameState;
  const numPlayers = players.length;

  const eliminationMessage = (() => {
    if (!currentEvent) {
      return '';
    }

    if (currentEvent.action_type === 'banish_result') {
      const eliminatedId = currentEvent.payload?.eliminated;
      return eliminatedId != null ? `P${eliminatedId} was banished` : 'A player was banished';
    }

    if (currentEvent.action_type === 'murder_result') {
      const eliminatedId = currentEvent.payload?.eliminated;
      return eliminatedId != null ? `P${eliminatedId} was murdered` : 'A player was murdered';
    }

    return '';
  })();

  const buildVoteLookup = () => {
    const lookup = new Map();

    const upsert = (voterId, value) => {
      const key = Number(voterId);
      const existing = lookup.get(key) || {};
      lookup.set(key, { ...existing, ...value });
    };

    (voteSummary?.banishVotes || []).forEach((row) => {
      upsert(row.voterId, { banishTarget: row.targetId });
    });

    // Revote supersedes main vote when present.
    (voteSummary?.revote || []).forEach((row) => {
      upsert(row.voterId, { banishTarget: row.targetId, usedRevote: true });
    });

    (voteSummary?.murderVotes || []).forEach((row) => {
      upsert(row.voterId, { murderTarget: row.targetId });
    });

    return lookup;
  };

  const voteLookup = buildVoteLookup();

  const roundEvents = (gameState.events || []).filter((e) => e.round === gameState.round);
  const banishResultSeen = roundEvents.some((e) => e.action_type === 'banish_result');
  const murderResultSeen = roundEvents.some((e) => e.action_type === 'murder_result');

  // After banish_result: hide regular votes, but murder votes can still appear.
  // After murder_result: hide murder votes too.
  const showOnlyMurderVotes =
    banishResultSeen &&
    !murderResultSeen &&
    (voteSummary?.murderVotes || []).length > 0;
  const hideBanishVotes = banishResultSeen;
  const hideMurderVotes = murderResultSeen;

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
              {eliminationMessage ? (
                <div className="max-w-[75%] rounded-xl border border-amber-300/40 bg-slate-950/75 px-4 py-3 text-center text-sm font-semibold text-amber-100 shadow-lg">
                  {eliminationMessage}
                </div>
              ) : (
                <div className="text-4xl opacity-20">🎭</div>
              )}
            </div>
          </div>
        </div>

        {players.map((player, index) => {
          const pos = getPlayerPosition(index);
          const playerVotes = voteLookup.get(player.id) || {};
          const hasVoteInfo = showOnlyMurderVotes
            ? playerVotes.murderTarget != null
            : !hideBanishVotes && (playerVotes.banishTarget != null || (!hideMurderVotes && playerVotes.murderTarget != null));


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
                <div
                  className="absolute -bottom-6 left-1/2 max-w-[9.5rem] whitespace-nowrap rounded bg-slate-800 px-2 py-1 text-xs font-bold text-white transform -translate-x-1/2"
                  title={`P${player.id} - ${player.displayName || `Player ${player.id}`}`}
                >
                  <span className="truncate inline-block max-w-[9rem] align-bottom">
                    P{player.id} · {player.displayName || `Player ${player.id}`}
                  </span>
                </div>
              </div>
              {hasVoteInfo && (
                <div className="mt-7 flex gap-1">
                  {!hideBanishVotes && !showOnlyMurderVotes && playerVotes.banishTarget != null && (
                    <span className="rounded bg-cyan-900/80 px-2 py-0.5 text-[10px] font-semibold text-cyan-100">
                      {playerVotes.usedRevote ? 'R' : 'V'}: P{playerVotes.banishTarget}
                    </span>
                  )}
                  {!hideMurderVotes && playerVotes.murderTarget != null && (
                    <span className="rounded bg-rose-900/80 px-2 py-0.5 text-[10px] font-semibold text-rose-100">
                      M: P{playerVotes.murderTarget}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

    </div>
  );
}

export default RoundTable;
