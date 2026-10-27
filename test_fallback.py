#!/usr/bin/env python3
"""
Test script to verify API key fallback mechanism.
Run this to check if your multiple API keys are properly configured.

Usage:
    python test_fallback.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.config import settings

def test_api_keys():
    """Test API key configuration and fallback setup."""
    
    print("=" * 60)
    print("🔑 API Key Fallback Configuration Test")
    print("=" * 60)
    print()
    
    # Check primary key
    print("1️⃣  Primary Key (GEMINI_API_KEY):")
    if settings.gemini_api_key:
        masked_key = settings.gemini_api_key[:15] + "..." + settings.gemini_api_key[-8:]
        print(f"   ✅ Configured: {masked_key}")
    else:
        print("   ❌ NOT configured")
    print()
    
    # Check secondary key
    print("2️⃣  Fallback Key #2 (GEMINI_API_KEY_2):")
    if settings.gemini_api_key_2:
        masked_key = settings.gemini_api_key_2[:15] + "..." + settings.gemini_api_key_2[-8:]
        print(f"   ✅ Configured: {masked_key}")
    else:
        print("   ⚠️  Not configured (optional)")
    print()
    
    # Check tertiary key
    print("3️⃣  Fallback Key #3 (GEMINI_API_KEY_3):")
    if settings.gemini_api_key_3:
        masked_key = settings.gemini_api_key_3[:15] + "..." + settings.gemini_api_key_3[-8:]
        print(f"   ✅ Configured: {masked_key}")
    else:
        print("   ⚠️  Not configured (optional)")
    print()
    
    # Get available keys
    available_keys = settings.get_available_gemini_keys()
    
    print("-" * 60)
    print(f"📊 Total Configured Keys: {len(available_keys)}")
    print("-" * 60)
    print()
    
    if len(available_keys) == 0:
        print("❌ ERROR: No API keys configured!")
        print()
        print("📝 Action Required:")
        print("   1. Edit backend/.env")
        print("   2. Add at least GEMINI_API_KEY=your_key_here")
        print("   3. Get API keys from: https://ai.google.dev/")
        return False
        
    elif len(available_keys) == 1:
        print("✅ Basic Setup Complete")
        print()
        print("💡 Recommendation:")
        print("   Add more keys to backend/.env for better reliability:")
        print("   GEMINI_API_KEY_2=your_second_key")
        print("   GEMINI_API_KEY_3=your_third_key")
        print()
        print("   Benefits:")
        print("   • 3x more API quota")
        print("   • Automatic failover on rate limits")
        print("   • Zero downtime during quota exhaustion")
        return True
        
    elif len(available_keys) == 2:
        print("✅ Good Setup!")
        print()
        print("💡 Optional Enhancement:")
        print("   Add GEMINI_API_KEY_3 for maximum reliability")
        return True
        
    else:  # 3 keys
        print("✅ Excellent! Maximum Reliability Setup")
        print()
        print("🎯 Your Configuration:")
        print("   • 3x API quota capacity")
        print("   • Automatic failover enabled")
        print("   • Maximum uptime reliability")
        return True
    
    print()

def test_fallback_function():
    """Test that the fallback function works."""
    print()
    print("=" * 60)
    print("🧪 Testing Fallback Function")
    print("=" * 60)
    print()
    
    try:
        available_keys = settings.get_available_gemini_keys()
        print(f"✅ get_available_gemini_keys() returned {len(available_keys)} key(s)")
        print()
        
        # Test each key format
        for i, key in enumerate(available_keys, 1):
            if key.startswith("AIza") or key.startswith("AQ."):
                print(f"✅ Key #{i}: Valid format")
            else:
                print(f"⚠️  Key #{i}: Unusual format (might still work)")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Error testing fallback function: {e}")
        return False

def test_other_settings():
    """Test other important settings."""
    print()
    print("=" * 60)
    print("⚙️  Other Configuration")
    print("=" * 60)
    print()
    
    print(f"📦 Embedding Model: {settings.gemini_embed_model}")
    print(f"🤖 Generation Model: {settings.gemini_gen_model}")
    print(f"💾 Cache Path: {settings.cache_db_path}")
    print(f"🔗 CORS Origins: {settings.cors_origins}")
    print(f"📊 Similarity Threshold: {settings.similarity_threshold}")
    print()
    
    if settings.pinecone_api_key:
        masked = settings.pinecone_api_key[:10] + "..." + settings.pinecone_api_key[-6:]
        print(f"🌲 Pinecone: ✅ Configured ({masked})")
    else:
        print(f"🌲 Pinecone: ⚠️  Not configured (optional)")
    print()

def main():
    """Run all tests."""
    try:
        success = test_api_keys()
        test_fallback_function()
        test_other_settings()
        
        print("=" * 60)
        print("🏁 Test Complete")
        print("=" * 60)
        print()
        
        if success:
            print("✅ Configuration looks good!")
            print()
            print("📝 Next Steps:")
            print("   1. Start backend: uvicorn app.api:app --reload --port 8000")
            print("   2. Check health: http://127.0.0.1:8000/api/health")
            print("   3. Monitor logs for fallback messages")
            print()
        else:
            print("❌ Please fix configuration issues above")
            print()
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
