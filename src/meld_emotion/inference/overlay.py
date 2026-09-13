"""The one-window overlay both demo frontends draw (design doc §8.1): face
boxes with track IDs from the last SAMPLED frame, the faces-only reading as
one label, a status word, and up to a few caption lines at the bottom
(transcript, final state, response)."""
import cv2

BOX_COLOUR, TEXT_COLOUR = (0, 255, 0), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, boxes: list, track_ids: list, face_label: str | None, lines: list[str],
                 status: str = "") -> None:
    for (x, y, w, h, score), track_id in zip(boxes, track_ids):
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOUR, 2)
        cv2.putText(frame, f"id{track_id}", (x, max(0, y - 6)), FONT, 0.5, BOX_COLOUR, 1)
    if face_label:
        cv2.putText(frame, face_label, (8, 24), FONT, 0.6, TEXT_COLOUR, 1)
    if status:
        (tw, _), _ = cv2.getTextSize(status, FONT, 0.55, 1)
        cv2.putText(frame, status, (frame.shape[1] - tw - 10, 22), FONT, 0.55, TEXT_COLOUR, 1)
    lines = [line for line in lines if line]
    if lines:
        h, row = frame.shape[0], 22
        cv2.rectangle(frame, (0, h - row * len(lines) - 6), (frame.shape[1], h), (0, 0, 0), -1)
        for i, line in enumerate(reversed(lines)):
            cv2.putText(frame, line[:110], (8, h - 8 - i * row), FONT, 0.5, TEXT_COLOUR, 1)


def face_label_for(faces_seen: int, vision_only_expression: dict | None, threshold: float) -> str:
    from meld_emotion.inference.gloss import face_reading
    if faces_seen <= 0 or not vision_only_expression:
        return "face: none"
    label, prob = face_reading(vision_only_expression, threshold)
    return f"face: {label} ({prob:.0%})"
