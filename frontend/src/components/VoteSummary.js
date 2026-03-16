import React from 'react';

function VoteRows({ rows, emptyLabel, accentClass }) {
  if (!rows.length) {
    return <div className="text-sm text-slate-500">{emptyLabel}</div>;
  }

  return (
    <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
      {rows.map((row) => (
        <div
          key={`${row.voterId}-${row.targetId}`}
          className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-800/70 px-3 py-2"
        >
          <span className="text-sm font-semibold text-slate-200">P{row.voterId}</span>
          <span className={`rounded-full px-2 py-1 text-xs font-bold ${accentClass}`}>
            P{row.targetId}
          </span>
        </div>
      ))}
    </div>
  );
}

function VoteCounts({ counts, label }) {
  const entries = Object.entries(counts).sort((left, right) => Number(left[0]) - Number(right[0]));

  if (!entries.length) {
    return null;
  }

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      <span className="text-xs uppercase tracking-[0.2em] text-slate-500">{label}</span>
      {entries.map(([targetId, count]) => (
        <span
          key={targetId}
          className="rounded-full border border-slate-700 bg-slate-950/80 px-2 py-1 text-xs font-semibold text-slate-300"
        >
          P{targetId}: {count}
        </span>
      ))}
    </div>
  );
}

function VoteSection({ title, subtitle, rows, counts, accentClass, emptyLabel }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h4 className="text-sm font-bold uppercase tracking-[0.25em] text-slate-300">{title}</h4>
          <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
        </div>
        <div className="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-400">
          {rows.length} shown
        </div>
      </div>
      <VoteRows rows={rows} emptyLabel={emptyLabel} accentClass={accentClass} />
      <VoteCounts counts={counts} label="Totals" />
    </div>
  );
}

function VoteSummary({ voteSummary, round }) {
  if (!voteSummary) {
    return null;
  }

  const hasVotes = voteSummary.banishVotes.length || voteSummary.revote.length || voteSummary.murderVotes.length;

  return (
    <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900 p-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="text-xl font-bold text-white">Vote Overview</h3>
          <p className="mt-1 text-sm text-slate-400">All visible votes for round {round} in one place.</p>
        </div>
        {!hasVotes && (
          <div className="rounded-full bg-slate-800 px-3 py-1 text-xs font-semibold text-slate-400">
            No votes yet
          </div>
        )}
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <VoteSection
          title="Banish Votes"
          subtitle="Main table vote before any tie-break."
          rows={voteSummary.banishVotes}
          counts={voteSummary.banishCounts}
          accentClass="bg-sky-500/15 text-sky-300"
          emptyLabel="No public banish votes are visible yet."
        />
        <VoteSection
          title="Revote"
          subtitle="Only appears if the first vote tied."
          rows={voteSummary.revote}
          counts={voteSummary.revoteCounts}
          accentClass="bg-amber-500/15 text-amber-300"
          emptyLabel="No revote happened in this round."
        />
      </div>

      <VoteSection
        title="Murder Votes"
        subtitle="Private traitor kill choices for this round."
        rows={voteSummary.murderVotes}
        counts={voteSummary.murderCounts}
        accentClass="bg-rose-500/15 text-rose-300"
        emptyLabel="No murder votes are visible yet."
      />
    </div>
  );
}

export default VoteSummary;