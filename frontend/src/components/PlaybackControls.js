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
  const speedOptions = [
    { label: '0.5x', value: 2000 },
    { label: '1x', value: 1000 },
    { label: '2x', value: 500 },
    { label: '4x', value: 250 }
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center space-x-4">
        {/* Previous */}
        <button
          onClick={onPrev}
          disabled={currentIndex === 0}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-900 disabled:text-gray-600 text-white rounded transition-colors"
        >
          ⏮️ Prev
        </button>

        {/* Play/Pause */}
        <button
          onClick={isPlaying ? onPause : onPlay}
          className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded font-semibold transition-colors"
        >
          {isPlaying ? '⏸️ Pause' : '▶️ Play'}
        </button>

        {/* Next */}
        <button
          onClick={onNext}
          disabled={currentIndex >= totalEvents - 1}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-900 disabled:text-gray-600 text-white rounded transition-colors"
        >
          Next ⏭️
        </button>

        {/* Progress bar */}
        <div className="flex-1">
          <input
            type="range"
            min="0"
            max={totalEvents - 1}
            value={currentIndex}
            onChange={(e) => onSeek(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div className="text-center text-sm text-gray-400 mt-1">
            Event {currentIndex + 1} / {totalEvents}
          </div>
        </div>

        {/* Speed controls */}
        <div className="flex space-x-1">
          {speedOptions.map(option => (
            <button
              key={option.value}
              onClick={() => onSpeedChange(option.value)}
              className={`
                px-3 py-2 rounded text-sm transition-colors
                ${playbackSpeed === option.value 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
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
