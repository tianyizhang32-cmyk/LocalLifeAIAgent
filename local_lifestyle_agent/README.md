Local Lifestyle Agent (Yelp-like) — MVP Skeleton

Credentials
-----------
This project uses TWO API keys (both are credentials / secrets):

1) OPENAI_API_KEY
   - Authenticates your requests to the OpenAI API (models like gpt-5.2).
   - Keep it secret; never commit.

2) GOOGLE_PLACES_API_KEY
   - Authenticates requests to Google Places API.
   - Keep it secret; never commit.

Local / Notebook usage
----------------------
Preferred: environment variables
  export OPENAI_API_KEY="..."
  export GOOGLE_PLACES_API_KEY="..."
  export OPENAI_MODEL="gpt-5.2"   # optional

Notebook-safe: interactive prompt
  from local_lifestyle_agent.config import Settings
  settings = Settings.load(interactive=True)
