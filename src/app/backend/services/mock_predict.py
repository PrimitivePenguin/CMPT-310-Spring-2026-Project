import random

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def mock_predict() -> dict:
    emotion = random.choice(EMOTIONS)
    confidence = round(random.uniform(0.65, 0.95), 2)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "source": "mock",
    }