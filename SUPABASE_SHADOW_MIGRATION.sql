-- Run ONCE in Supabase -> SQL Editor.  Adds the table that stores "shadow" (paper) trades:
-- setups the engine blocked, tracked virtually (no money) so it can learn whether the block was right.
-- Without this table the feature still works, but shadow data is kept only in the app's local file and is lost on restart.
create table if not exists public.ai_shadow_setups (
  id bigserial primary key,
  timestamp text not null,
  direction text not null,
  underlying_entry double precision not null,
  stop_loss double precision not null,
  target double precision not null,
  blocked_by text not null default 'Other',
  factors_json text,
  status text not null default 'OPEN' check (status in ('OPEN','WIN','LOSS','EXPIRED')),
  exit_price double precision,
  exit_timestamp text
);
create index if not exists idx_ai_shadow_status_id on public.ai_shadow_setups(status, id desc);
alter table public.ai_shadow_setups enable row level security;
-- same access style as ai_trade_setups: allow the app's key to read/write
do $$ begin
  if not exists (select 1 from pg_policies where tablename='ai_shadow_setups' and policyname='allow_all_shadow') then
    create policy allow_all_shadow on public.ai_shadow_setups for all using (true) with check (true);
  end if;
end $$;
