import base64
import mimetypes
from pathlib import Path


def contains_image_inputs(messages):
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("image_path") or msg.get("image_paths"):
            return True
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") in ("image_url", "image")
                ):
                    return True
    return False


def encode_image(image_path):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/jpeg"

    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    return f"data:{mime};base64,{data}"


def normalize_messages_with_images(messages):
    normalized = []

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("Each message must be a dict.")

        content = msg.get("content")
        image_paths = []

        if msg.get("image_path"):
            image_paths.append(msg["image_path"])
        if msg.get("image_paths"):
            image_paths.extend(msg["image_paths"])

        if isinstance(content, list):
            parts = list(content)
        else:
            parts = []
            if content:
                parts.append({"type": "text", "text": str(content)})

        if image_paths:
            for image_path in image_paths:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encode_image(image_path)
                        },
                    }
                )

        if image_paths or isinstance(content, list):
            new_msg = {
                k: v
                for k, v in msg.items()
                if k not in ("image_path", "image_paths")
            }
            new_msg["content"] = parts
            normalized.append(new_msg)
        else:
            normalized.append(msg)

    return normalized
