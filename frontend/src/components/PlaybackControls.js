import React from 'react';

function PlaybackControls({
  currentIndex,
  totalEvents,
  isPlaying,
  playbackSpeed,
  onPlay,
  onPause,
  onNext,
  onPrev,
  onSeek,
  onSpeedChange
}) {
  const safeMax = Math.max(totalEvents - 1, 0);

  const speedOptions = [
    { label: '0.5x', value: 2000 },
    { label: '1x', value: 1000 },
    { label: '2x', value: 500 },
    { label: '4x', value: 250 }
  ];

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4">
      <div className="flex items-center space-x-4">
        <button
          onClick={onPrev}
          disabled={currentIndex === 0}
          className="rounded bg-slate-800 px-4 py-2 text-white transition-colors hover:bg-slate-700 disabled:bg-slate-950 disabled:text-slate-600"
        >
          ⏮️ Prev
        </button>

        <button
          onClick={isPlaying ? onPause : onPlay}
          disabled={totalEvents < 2}
          className="rounded bg-sky-600 px-6 py-2 font-semibold text-white transition-colors hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-500"
        >
          {isPlaying ? '⏸️ Pause' : '▶️ Play'}
        </button>

        <button
          onClick={onNext}
          disabled={currentIndex >= totalEvents - 1}
          className="rounded bg-slate-800 px-4 py-2 text-white transition-colors hover:bg-slate-700 disabled:bg-slate-950 disabled:text-slate-600"
        >
          Next ⏭️
        </button>

        <div className="flex-1">
          <input
            type="range"
            min="0"
            max={safeMax}
            value={Math.min(currentIndex, safeMax)}
            onChange={(e) => onSeek(parseInt(e.target.value))}
            className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-700 accent-sky-600"
          />
          <div className="mt-1 text-center text-sm text-slate-400">
            Event {totalEvents === 0 ? 0 : currentIndex + 1} / {totalEvents}
          </div>
        </div>

        <div className="flex space-x-1">
          {speedOptions.map(option => (
            <button
              key={option.value}
              onClick={() => onSpeedChange(option.value)}
              className={`
                px-3 py-2 rounded text-sm transition-colors
                ${playbackSpeed === option.value
                  ? 'bg-sky-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }
              `}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default PlaybackControls;
