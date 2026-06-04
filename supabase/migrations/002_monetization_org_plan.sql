alter table organizations
add column if not exists plan text default 'free',
add column if not exists razorpay_customer_id text,
add column if not exists razorpay_subscription_id text;

update organizations
set plan = 'free'
where plan is null;
