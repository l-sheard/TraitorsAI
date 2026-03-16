import React, { useEffect, useRef } from 'react';

function TraitorChatPanel({ events, currentRound, maxHeight = '220px' }) {
  const scrollerRef = useRef(null);

  const messages = (events || []).filter(
    (event) =>
      (event.action_type === 'traitor_chat' && event.phase === 'traitor_chat') ||
      (event.action_type === 'murder' && event.phase === 'murder')
  );

  useEffect(() => {
    if (!scrollerRef.current) {
      return;
    }
    scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
  }, [messages.length]);

  return (
    <div className="rounded-2xl border border-rose-900/60 bg-rose-950/30 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xl font-bold text-rose-100">Traitor Private Chat</h3>
        <span className="text-xs uppercase tracking-wider text-rose-300/80">
          Through round {currentRound || 0}
        </span>
      </div>

      {messages.length === 0 ? (
        <p className="text-sm text-rose-200/80">No private traitor messages up to this point.</p>
      ) : (
        <div ref={scrollerRef} className="space-y-2" style={{ maxHeight, overflowY: 'auto' }}>
          {messages.map((event, index) => {
            const isMurderVote = event.action_type === 'murder';
            const content = isMurderVote
              ? `Murder vote for P${event.payload?.target_id ?? '?'}${event.payload?.rationale ? ` - ${event.payload.rationale}` : ''}`
              : event.payload?.content || '';

            return (
              <div key={`${event.round}-${event.actor_id}-${index}`} className="rounded-lg bg-slate-900/50 p-3">
                <div className="mb-1 text-xs text-rose-300">
                  Round {event.round} · P{event.actor_id}{isMurderVote ? ' - Murder Vote' : ''}
                </div>
                <div className="text-sm text-rose-100">{content}</div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default TraitorChatPanel;
