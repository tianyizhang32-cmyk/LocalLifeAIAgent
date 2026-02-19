# Security Guidelines

## API Keys and Secrets

**IMPORTANT**: Never commit API keys or secrets to the repository!

### Required API Keys

This project requires the following API keys:

1. **OpenAI API Key** - For LLM functionality
2. **Google Places API Key** - For venue search

### How to Set Up API Keys

#### Option 1: Environment Variables (Recommended)

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export GOOGLE_PLACES_API_KEY="your-google-places-api-key-here"
```

#### Option 2: Configuration File (Local Only)

1. Copy the example configuration:
   ```bash
   cp config.example.json config.json
   ```

2. Edit `config.json` and add your API keys:
   ```json
   {
     "api": {
       "openai": {
         "api_key": "your-openai-api-key-here"
       },
       "google_places": {
         "api_key": "your-google-places-api-key-here"
       }
     }
   }
   ```

3. **NEVER commit `config.json`** - it's already in `.gitignore`

### Files to Never Commit

- `config.json` (contains real API keys)
- `.env` files (contains environment variables)
- Any file with `.key` or `.pem` extension
- Any file in `output/` directory (may contain sensitive data)

### What's Safe to Commit

- `config.example.json` (template with placeholder values)
- All source code files
- Documentation
- Requirements files

### If You Accidentally Commit API Keys

1. **Immediately revoke the exposed keys** from your API provider
2. Generate new API keys
3. Remove the keys from Git history:
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch <file-with-keys>" \
     --prune-empty --tag-name-filter cat -- --all
   ```
4. Force push to remote (if already pushed):
   ```bash
   git push origin --force --all
   ```

### Reporting Security Issues

If you discover a security vulnerability, please email: [your-email@example.com]

Do NOT create a public GitHub issue for security vulnerabilities.
