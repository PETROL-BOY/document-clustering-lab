# Gemini API Key Fallback Mechanism

## Overview

This project now supports **multiple Gemini API keys with automatic fallback**. If the primary API key hits rate limits or quota exhaustion (HTTP 429 errors), the system automatically switches to backup keys.

## Configuration

### Setting Up Multiple API Keys

Edit `backend/.env` and add up to 3 API keys:

```env
# Primary API key - will be tried first
GEMINI_API_KEY=your_primary_api_key_here

# Fallback API key 2 - used if primary fails with rate limit or quota errors
GEMINI_API_KEY_2=your_secondary_api_key_here

# Fallback API key 3 - used if both primary and secondary fail
GEMINI_API_KEY_3=your_tertiary_api_key_here
```

### How It Works

1. **Primary Key First**: The system always attempts to use `GEMINI_API_KEY` first
2. **Automatic Detection**: When a 429 (rate limit) or quota exhaustion error occurs, the system detects it
3. **Retry Logic**: For each key, the system retries with exponential backoff (10s, 20s, 40s, etc.)
4. **Key Rotation**: After exhausting retries on one key, it automatically switches to the next available key
5. **Logging**: The system logs which key is being used, making it easy to track failovers

## What Gets Protected

The fallback mechanism is implemented across all Gemini API operations:

### 1. Document Embeddings
- **Function**: `embed_chunks_gemini()`
- **Used for**: Converting document chunks into vector representations
- **Fallback**: Automatic key rotation on 429 errors

### 2. Query Embeddings  
- **Function**: `embed_query_gemini()`
- **Used for**: Converting user queries into vectors for similarity search
- **Fallback**: Automatic key rotation on 429 errors

### 3. Cluster Theme Generation
- **Function**: `generate_theme_llm()`
- **Used for**: Generating cluster labels, summaries, and keywords
- **Fallback**: Tries all available keys before failing

### 4. Q&A / RAG System
- **Function**: `search_and_rag()`
- **Used for**: Answering user questions based on document corpus
- **Fallback**: Tries all available keys before returning error message

## Monitoring & Logs

Watch the backend console for fallback messages:

```
[embed] 429 quota hit on key #1 — waiting 10.0s before retry 1/6
[embed] API key #1 exhausted, switching to fallback key #2
[embed] Successfully switched to fallback API key #2
```

For content generation:
```
[generate] 429 quota hit on key #1 — waiting 5s before retry 1/3
[generate] API key #1 exhausted, trying next key...
[generate] Successfully used fallback API key #2
```

For RAG queries:
```
[RAG] API key #1 exhausted, trying next key...
[RAG] Successfully used fallback API key #2
```

## Health Check

Check the API health endpoint to see how many keys are configured:

```bash
curl http://127.0.0.1:8000/api/health
```

Response includes:
```json
{
  "status": "healthy",
  "gemini_configured": true,
  "gemini_keys_count": 3,
  "pinecone_configured": true,
  ...
}
```

## Best Practices

1. **Multiple Google Accounts**: Create API keys from different Google Cloud accounts to maximize quota
2. **Free Tier Limits**: Each free tier key has ~60 requests per minute
3. **Paid Accounts**: If using paid keys, you can significantly increase throughput
4. **Monitor Usage**: Watch the console logs to see which keys are being used most
5. **Stagger Usage**: The system automatically throttles requests to avoid rapid quota depletion

## Retry Strategy

### Embeddings
- **Retries per key**: 6 attempts with exponential backoff
- **Initial delay**: 10 seconds
- **Max delay**: 120 seconds (2 minutes)
- **Total keys tried**: All available keys (up to 3)

### Content Generation (Themes & Q&A)
- **Retries per key**: 3 attempts
- **Delay**: 5s, 10s, 15s
- **Total keys tried**: All available keys (up to 3)

## Error Handling

If all API keys are exhausted:

1. **Embeddings**: Raises `RuntimeError: "All API keys exhausted"`
2. **Theme Generation**: Returns error cluster with message
3. **Q&A**: Returns error message in the answer field

## Migration Guide

### From Single Key Setup

If you were using:
```env
GEMINI_API_KEY=your_api_key_here
```

Simply add the fallback keys:
```env
GEMINI_API_KEY=your_primary_key
GEMINI_API_KEY_2=your_fallback_key_1
GEMINI_API_KEY_3=your_fallback_key_2
```

No code changes needed - the system automatically detects and uses all configured keys.

## Obtaining Additional API Keys

1. Go to [Google AI Studio](https://ai.google.dev/)
2. Sign in with a different Google account (if you want separate quotas)
3. Click "Get API Key"
4. Copy the key and add it to your `.env` file
5. Restart the backend server

## Troubleshooting

### Still Getting 429 Errors

- **Check logs**: Verify all keys are being tried
- **Wait period**: Free tier resets hourly - wait before retrying
- **Upgrade**: Consider upgrading to paid API access

### Keys Not Being Used

- **Restart required**: After adding new keys, restart the backend server
- **Check format**: Ensure keys are on separate lines in `.env`
- **Verify keys**: Test each key individually in Google AI Studio

### Performance Impact

- **Minimal overhead**: Fallback logic only activates on errors
- **Caching**: Local SQLite cache reduces API calls significantly
- **Smart throttling**: 5-second delays between embedding batches

## Technical Implementation

Key files modified:
- `backend/.env` - Configuration with 3 key slots
- `backend/app/config.py` - Settings class with `get_available_gemini_keys()` method
- `backend/app/pipeline.py` - Fallback logic in all Gemini API functions
- `backend/app/api.py` - Health check updated with key count

## Support

For issues or questions about the fallback mechanism, check:
1. Backend console logs for detailed error messages
2. Health endpoint for configuration status
3. This documentation for configuration examples
