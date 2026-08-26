"""Generate short Doubao TTS prompts as MP3 files.

Set VOLC_APP_ID and VOLC_ACCESS_TOKEN in the environment first. The voice_type
must be copied from the Volcengine console; the name "Taiwan Xiaohe" may refer
to a private/legacy voice and cannot be guessed reliably.
"""

import argparse
import base64
import json
import os
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


ENDPOINT = "https://openspeech.bytedance.com/api/v1/tts"


def synthesize(text: str, output: Path, app_id: str, token: str, voice_type: str) -> None:
    payload = {
        "app": {"appid": app_id, "token": token, "cluster": "volcano_tts"},
        "user": {"uid": "xiaozhi"},
        "audio": {
            "voice_type": voice_type,
            "encoding": "mp3",
            "rate": 24000,
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": "plain",
            "operation": "query",
        },
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer;{token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    if result.get("code") != 3000 or not result.get("data"):
        raise RuntimeError(f"Doubao TTS failed: {result}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(result["data"]))
    print(f"saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="append", help="text to synthesize; may be repeated")
    parser.add_argument("--out-dir", default="tts", type=Path)
    parser.add_argument(
        "--voice-type",
        default="zh_female_xiaohe_uranus_bigtts",
        help="copy the exact voice_type from the Volcengine console",
    )
    args = parser.parse_args()

    app_id = os.environ.get("VOLC_APP_ID")
    token = os.environ.get("VOLC_ACCESS_TOKEN")
    if not app_id or not token:
        raise SystemExit("Please set VOLC_APP_ID and VOLC_ACCESS_TOKEN first")

    texts = args.text or [
        "心率过高，请注意休息",
        "血压偏高，请注意休息",
        "测量完成",
    ]
    for index, text in enumerate(texts, 1):
        synthesize(text, args.out_dir / f"prompt_{index:02d}.mp3", app_id, token, args.voice_type)


if __name__ == "__main__":
    main()
