-- ============================================================
-- AI Data Analyzer — Initial Supabase Schema
-- Run once in Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- ============================================================

-- ── 1. organizations ────────────────────────────────────────
create table if not exists public.organizations (
    id          uuid primary key default gen_random_uuid(),
    name        text not null,
    created_at  timestamptz not null default now()
);

-- ── 2. analysis_runs ────────────────────────────────────────
create table if not exists public.analysis_runs (
    id           uuid primary key default gen_random_uuid(),
    org_id       uuid references public.organizations(id) on delete cascade,
    user_id      uuid,
    dataset_name text,
    status       text,
    output_path  text,
    created_at   timestamptz not null default now()
);

-- ── 3. session_store ────────────────────────────────────────
create table if not exists public.session_store (
    token      text primary key,
    user_data  jsonb,
    expiry     timestamptz
);

-- ── Indexes for common queries ───────────────────────────────
create index if not exists idx_analysis_runs_org_id
    on public.analysis_runs(org_id);

create index if not exists idx_analysis_runs_created_at
    on public.analysis_runs(created_at desc);

create index if not exists idx_session_store_expiry
    on public.session_store(expiry);

-- ── Row Level Security ───────────────────────────────────────
-- Enable RLS on all tables
alter table public.organizations  enable row level security;
alter table public.analysis_runs  enable row level security;
alter table public.session_store  enable row level security;

-- Service role bypass (used by backend with SUPABASE_SERVICE_KEY)
-- This allows all operations from the server side.
-- Drop existing policies first to be idempotent.
drop policy if exists "service_role_all_orgs"    on public.organizations;
drop policy if exists "service_role_all_runs"    on public.analysis_runs;
drop policy if exists "service_role_all_sessions" on public.session_store;

create policy "service_role_all_orgs"
    on public.organizations for all
    using (true)
    with check (true);

create policy "service_role_all_runs"
    on public.analysis_runs for all
    using (true)
    with check (true);

create policy "service_role_all_sessions"
    on public.session_store for all
    using (true)
    with check (true);
