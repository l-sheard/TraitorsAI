import React, { useEffect, useMemo, useState } from 'react';

import { fetchReplay } from '../api/replayApi';
import { buildReplaySnapshot, getWinnerPresentation } from '../lib/replayState';
import RoundTable from './RoundTable';
import EventLog from './EventLog';
import PlaybackControls from './PlaybackControls';
import TraitorChatPanel from './TraitorChatPanel';
import PlayerSpeechPanel from './PlayerSpeechPanel';

function trimEventsToFirstDiscussion(events) {
  if (!Array.isArray(events) || events.length === 0) {
    return [];
  }

  const firstDiscussionIndex = events.findIndex(
    (event) => event.phase === 'discussion' || event.action_type === 'public_message'
  );

  if (firstDiscussionIndex < 0) {
    return events;
  }

  return events.slice(firstDiscussionIndex);
}

function GameViewer({ gameId, onBack }) {
  const [summary, setSummary] = useState(null);
  const [events, setEvents] = useState([]);
  const [currentEventIndex, setCurrentEventIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadGameData();
  }, [gameId]);

  useEffect(() => {
    let interval;
    if (isPlaying && currentEventIndex < events.length - 1) {
      interval = setInterval(() => {
        setCurrentEventIndex(prev => {
          if (prev >= events.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentEventIndex, events.length, playbackSpeed]);

  const loadGameData = async () => {
    try {
      setLoading(true);
      setError('');
      setCurrentEventIndex(0);
      setIsPlaying(false);
      const replay = await fetchReplay(gameId);
      setSummary(replay.summary);
      setEvents(trimEventsToFirstDiscussion(replay.events));
    } catch (error) {
      console.error('Error loading game data:', error);
      setError('Unable to load this replay.');
    } finally {
      setLoading(false);
    }
  };

  const replaySnapshot = useMemo(
    () => buildReplaySnapshot(summary, events, currentEventIndex),
    [summary, events, currentEventIndex]
  );

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-white text-xl">Loading game...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-rose-900/60 bg-rose-950/40 p-8 text-center">
        <p className="text-lg text-rose-200">{error}</p>
        <button
          type="button"
          onClick={onBack}
          className="mt-4 rounded-lg bg-slate-800 px-4 py-2 font-semibold text-white transition hover:bg-slate-700"
        >
          Back to games
        </button>
      </div>
    );
  }

  const winnerPresentation = getWinnerPresentation(summary?.winner);

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <button
              onClick={onBack}
              className="text-sm text-slate-400 transition-colors hover:text-white"
            >
              ← Back to Games
            </button>
            <h2 className="mt-2 text-2xl font-bold text-white">{gameId}</h2>
            <p className="mt-1 text-sm text-slate-400">
              Round {replaySnapshot.round} · {replaySnapshot.phase}
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl bg-slate-800 px-4 py-3">
              <div className="text-xs uppercase tracking-wider text-slate-400">Winner</div>
              <div className={`mt-1 font-semibold ${winnerPresentation.className}`}>{winnerPresentation.label}</div>
            </div>
            <div className="rounded-xl bg-slate-800 px-4 py-3">
              <div className="text-xs uppercase tracking-wider text-slate-400">Configuration</div>
              <div className="mt-1 text-sm text-slate-100">{summary?.config?.n_players} players · {summary?.config?.n_traitors} traitors</div>
            </div>
            <div className="rounded-xl bg-slate-800 px-4 py-3">
              <div className="text-xs uppercase tracking-wider text-slate-400">Event index</div>
              <div className="mt-1 text-sm text-slate-100">{events.length === 0 ? 0 : currentEventIndex + 1} / {events.length}</div>
            </div>
          </div>
        </div>
      </div>

      <PlaybackControls
        currentIndex={currentEventIndex}
        totalEvents={events.length}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
        onPlay={() => setIsPlaying(events.length > 1)}
        onPause={() => setIsPlaying(false)}
        onNext={() => setCurrentEventIndex(Math.min(currentEventIndex + 1, events.length - 1))}
        onPrev={() => setCurrentEventIndex(Math.max(currentEventIndex - 1, 0))}
        onSeek={setCurrentEventIndex}
        onSpeedChange={setPlaybackSpeed}
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-start">
        <div className="lg:col-span-6">
          <RoundTable gameState={replaySnapshot} minHeight="420px" />
        </div>

        <div className="lg:col-span-6">
          <div className="space-y-4">
            <PlayerSpeechPanel
              events={replaySnapshot.events}
              currentRound={replaySnapshot.round}
              maxHeight="260px"
            />
            <TraitorChatPanel
              events={replaySnapshot.events}
              currentRound={replaySnapshot.round}
              maxHeight="260px"
            />
          </div>
        </div>
      </div>

      <div>
        <EventLog
          events={events}
          currentIndex={currentEventIndex}
          onSelectEvent={setCurrentEventIndex}
          maxHeight="340px"
        />
      </div>
    </div>
  );
}

export default GameViewer;
