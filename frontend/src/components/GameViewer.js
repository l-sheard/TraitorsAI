import React, { useState, useEffect } from 'react';
import axios from 'axios';
import RoundTable from './RoundTable';
import EventLog from './EventLog';
import PlaybackControls from './PlaybackControls';

const API_BASE = 'http://localhost:8000';

function GameViewer({ gameId, onBack }) {
  const [summary, setSummary] = useState(null);
  const [events, setEvents] = useState([]);
  const [currentEventIndex, setCurrentEventIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1000); // ms between events
  const [loading, setLoading] = useState(true);

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
      const [summaryRes, eventsRes] = await Promise.all([
        axios.get(`${API_BASE}/games/${gameId}/summary`),
        axios.get(`${API_BASE}/games/${gameId}/events`)
      ]);
      setSummary(summaryRes.data);
      setEvents(eventsRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Error loading game data:', error);
      setLoading(false);
    }
  };

  const getCurrentState = () => {
    // Build game state up to current event
    const eventsUpToCurrent = events.slice(0, currentEventIndex + 1);
    return {
      events: eventsUpToCurrent,
      currentEvent: events[currentEventIndex],
      summary
    };
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-white text-xl">Loading game...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-800 rounded-lg p-4 flex justify-between items-center">
        <button
          onClick={onBack}
          className="text-gray-400 hover:text-white transition-colors"
        >
          ← Back to Games
        </button>
        <div className="text-white text-center">
          <h2 className="text-xl font-bold">{gameId}</h2>
          <p className="text-sm text-gray-400">
            Round {events[currentEventIndex]?.round || 0} - {events[currentEventIndex]?.phase || 'start'}
          </p>
        </div>
        <div className={`text-lg font-bold ${
          summary?.winner === 'traitor' ? 'text-red-500' : 'text-blue-500'
        }`}>
          {summary?.winner === 'traitor' ? '🗡️ Traitors Win' : '🛡️ Faithful Win'}
        </div>
      </div>

      {/* Main View */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Round Table - takes up 2 columns */}
        <div className="lg:col-span-2">
          <RoundTable gameState={getCurrentState()} />
        </div>

        {/* Event Log - takes up 1 column */}
        <div className="lg:col-span-1">
          <EventLog 
            events={events} 
            currentIndex={currentEventIndex}
            onSelectEvent={setCurrentEventIndex}
          />
        </div>
      </div>

      {/* Playback Controls */}
      <PlaybackControls
        currentIndex={currentEventIndex}
        totalEvents={events.length}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onNext={() => setCurrentEventIndex(Math.min(currentEventIndex + 1, events.length - 1))}
        onPrev={() => setCurrentEventIndex(Math.max(currentEventIndex - 1, 0))}
        onSeek={setCurrentEventIndex}
        onSpeedChange={setPlaybackSpeed}
      />
    </div>
  );
}

export default GameViewer;
