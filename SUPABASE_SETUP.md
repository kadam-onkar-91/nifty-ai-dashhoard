# Supabase persistent learning setup

1. Create a Supabase project.
2. Open **SQL Editor** and run `supabase_schema.sql`.
3. In Streamlit Cloud, open **App → Settings → Secrets** and add:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_SERVER_SIDE_KEY"
```

4. Redeploy the app.
5. The dashboard should show the green persistent-storage message when the connection works.

The code keeps SQLite as a fallback. If Supabase is unavailable/not configured,
the learner falls back to local SQLite and the dashboard warns that persistence
is not guaranteed on ephemeral deployments such as Streamlit Cloud.
