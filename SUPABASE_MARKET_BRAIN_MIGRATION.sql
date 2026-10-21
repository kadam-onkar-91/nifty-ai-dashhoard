-- Run ONCE in Supabase -> SQL Editor.  Adds the two tables used by the Market Brain (v46):
--   market_snapshots : one row per completed 5-min candle (features + the 60-min outcome label + the forecast made live)
--   market_autopsy   : one row per finished trading day (swings, reasons for each reversal, which big moves the engine caught)
-- Without these tables the Market Brain still works, but keeps its data only in the app's local file (lost on restart;
-- the price-based part is rebuilt from the candles, the live OI/PCR/VIX/breadth part is not).
create table if not exists public.market_snapshots (
  ts text primary key,            -- candle open time, IST, 'YYYY-MM-DD HH:MM:SS'
  features text,                  -- JSON
  labels text,                    -- JSON, filled once the next 12 candles exist
  src text not null default 'backfill',   -- 'live' (recorded while the app was running) or 'backfill'
  forecast text                   -- JSON, the forecast made at that candle (live rows only)
);
create table if not exists public.market_autopsy (
  day text primary key,           -- 'YYYY-MM-DD'
  summary text                    -- JSON
);
alter table public.market_snapshots enable row level security;
alter table public.market_autopsy enable row level security;
do $$ begin
  if not exists (select 1 from pg_policies where tablename='market_snapshots' and policyname='allow_all_market_snapshots') then
    create policy allow_all_market_snapshots on public.market_snapshots for all using (true) with check (true);
  end if;
  if not exists (select 1 from pg_policies where tablename='market_autopsy' and policyname='allow_all_market_autopsy') then
    create policy allow_all_market_autopsy on public.market_autopsy for all using (true) with check (true);
  end if;
end $$;
