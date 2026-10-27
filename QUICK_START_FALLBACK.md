# Quick Start: Multiple API Keys Setup

## 🎯 Goal
Set up multiple Gemini API keys to avoid rate limit errors (429).

## 📋 Steps

### 1. Get Additional API Keys (5 minutes)

Visit: **https://ai.google.dev/**

For each key you want:
- Use a different Google account (recommended for separate quotas)
- Click "Get API Key" 
- Copy the generated key

### 2. Add Keys to `.env` (1 minute)

Open: `backend/.env`

Add your keys:
```env
# Primary (already configured)
GEMINI_API_KEY=AIzaSy...existing_key

# Add these two lines:
GEMINI_API_KEY_2=AIzaSy...second_key_here
GEMINI_API_KEY_3=AIzaSy...third_key_here
```

### 3. Restart Backend (30 seconds)

In your backend terminal:
1. Press **Ctrl+C** to stop
2. Run: `uvicorn app.api:app --reload --port 8000`

### 4. Verify (10 seconds)

Visit: http://127.0.0.1:8000/api/health

Look for:
```json
{
  "gemini_keys_count": 3,  ← Should show 3
  ...
}
```

## ✅ Done!

Your application now has **3x the API quota** and will automatically failover when hitting rate limits.

## 🔍 How to Tell It's Working

Watch your backend console. When key #1 hits a limit, you'll see:

```
[embed] 429 quota hit on key #1 — waiting 10.0s before retry 1/6
[embed] API key #1 exhausted, switching to fallback key #2
[embed] Successfully switched to fallback API key #2
```

## 💡 Tips

- **Optional**: You only need 1 key minimum (what you have now)
- **Better**: 2-3 keys give you more quota and reliability
- **Free Tier**: Each key gets ~60 requests/minute
- **Paid**: Upgrade any key at https://console.cloud.google.com/

## 📞 Troubleshooting

**Q: Still getting 429 errors?**  
A: Check you restarted the backend after adding keys

**Q: How do I know which keys are working?**  
A: Check `/api/health` endpoint - shows `gemini_keys_count`

**Q: Can I use just 2 keys instead of 3?**  
A: Yes! Just fill in `GEMINI_API_KEY` and `GEMINI_API_KEY_2`

**Q: Do I need different Google accounts?**  
A: No, but different accounts = separate quotas = more capacity

---

**Status**: Your primary key is working now. Add backups when ready! 🚀
