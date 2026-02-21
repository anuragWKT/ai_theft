# ai_theft

## Start App (LiveKit Room)

1. Activate environment:
	- `source venv/bin/activate`
2. env fill keys:
	- Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_ROOM`, `LIVEKIT_IDENTITY`
	- Set `VOICE_STT_API_KEY`, `VOICE_TTS_API_KEY`
3. Start runtime:
	- `python -m src.livekit_app.voice_runtime`

## Generate Join Token

Run:

`source venv/bin/activate && python -c "from livekit import api; print(api.AccessToken('devkey','devsecret').with_identity('anurag-user').with_name('anurag-user').with_grants(api.VideoGrants(room_join=True, room='birthday-interrogation', can_publish=True, can_subscribe=True)).to_jwt())"`

## Where To Connect

1. Open `https://meet.livekit.io`
2. Server URL: `ws://localhost:7880`
3. Paste token from previous step
4. Join room and allow mic/speaker permissions

## How To Play

- Speak naturally in the room microphone.
- Ask timeline questions to build Alex's story.
	- Example: `What time did you arrive?`
	- Example: `What did you do after 7:00 PM?`
- Use camera checks to verify claims.
	- Example: `camera 8:00 pm`
	- Example: `camera 9:30 pm`
- Interrogate for contradictions and then accuse with time + evidence.
	- Example: `accuse 6:00 pm camera shows chain visible`

### Win / Lose Behavior

- **Win condition**: Correct theft time + sufficient evidence in accusation.
- On win, Alex confesses and the game session ends.
- **False/weak accusation**: Alex denies and game continues.

### Help

- Focus on timeline questions, camera checks, and evidence-based accusations.
- Runtime prints live game status in terminal each turn.

### Tips

- Lock Alex's timeline first across multiple times.
- Verify each claim with camera checks at the same times.
- Re-ask key times in different wording to expose contradictions.
- Accuse only when you have time + camera-supported evidence.
- Strong example: `accuse 6:00 pm camera shows chain visible`