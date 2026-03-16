import React, { useEffect, useMemo, useRef } from 'react';

function PlayerSpeechPanel({ events, currentRound, maxHeight = '280px' }) {
  const scrollerRef = useRef(null);

  const roundMessages = useMemo(
    () =>
      (events || []).filter(
        (event) => {
          const isDiscussionMessage =
            event.action_type === 'public_message' && event.phase === 'discussion';
          const isVoteMessage =
            event.action_type === 'vote' && (event.phase === 'voting' || event.phase === 'revote');

          return (isDiscussionMessage || isVoteMessage) && event.round === currentRound;
        }
      ),
    [events, currentRound]
  );

  useEffect(() => {
    if (!scrollerRef.current) {
      return;
    }
    scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
  }, [roundMessages.length]);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xl font-bold text-white">Player Speech</h3>
        <span className="text-xs uppercase tracking-wider text-slate-400">
          Round {currentRound || 0}
        </span>
      </div>

      <div ref={scrollerRef} className="space-y-2" style={{ maxHeight, overflowY: 'auto' }}>
        {roundMessages.length === 0 ? (
          <p className="text-sm text-slate-500">No message yet this round.</p>
        ) : (
          roundMessages.map((event, index) => {
            const isVote = event.action_type === 'vote';
            const body = isVote
              ? `Votes for P${event.payload?.target_id ?? '?'}${event.payload?.rationale ? ` - ${event.payload.rationale}` : ''}`
              : event.payload?.content || '';

            return (
              <div key={`${event.round}-${event.actor_id}-${index}`} className="rounded-lg border border-slate-800 bg-slate-800/70 p-3">
                <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  P{event.actor_id}{isVote ? ' - Vote' : ''}
                </div>
                <div className="text-sm text-slate-100">{body}</div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

export default PlayerSpeechPanel;
