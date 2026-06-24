-- ============================================================
-- AI Data Analyzer — Add org_members table for isolation
-- ============================================================

create table if not exists public.org_members (
    org_id uuid references public.organizations(id) on delete cascade,
    user_id uuid not null,
    role text default 'member',
    created_at timestamptz not null default now(),
    primary key (org_id, user_id)
);

create index if not exists idx_org_members_user_id on public.org_members(user_id);

alter table public.org_members enable row level security;

-- Service role bypass
drop policy if exists "service_role_all_org_members" on public.org_members;
create policy "service_role_all_org_members"
    on public.org_members for all
    to service_role
    using (true)
    with check (true);

grant all on public.org_members to service_role;
