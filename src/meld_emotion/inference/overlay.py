"""The one-window overlay both demo frontends draw (design doc §8.1): face
boxes with track IDs from the last SAMPLED frame, a bar per emotion for the
provisional (faces-only, text-masked) state, a status word, and up to a
few caption lines at the bottom (transcript, final state, response)."""
import cv2

BAR_COLOUR, BOX_COLOUR, TEXT_COLOUR = (200, 200, 0), (0, 255, 0), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(frame, boxes: list, track_ids: list, provisional: dict | None, lines: list[str],
                 status: str = "") -> None:
    for (x, y, w, h, score), track_id in zip(boxes, track_ids):
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOUR, 2)
        cv2.putText(frame, f"id{track_id}", (x, max(0, y - 6)), FONT, 0.5, BOX_COLOUR, 1)
    if provisional:
        bar_x, bar_y, bar_w, row_h = 8, 22, 100, 14
        cv2.putText(frame, "faces only", (bar_x, bar_y - 8), FONT, 0.4, BAR_COLOUR, 1)
        for i, (label, prob) in enumerate(provisional.items()):
            y = bar_y + i * row_h
            cv2.rectangle(frame, (bar_x, y), (bar_x + int(bar_w * prob), y + row_h - 4), BAR_COLOUR, -1)
            cv2.putText(frame, label[:4], (bar_x + bar_w + 4, y + row_h - 6), FONT, 0.4, TEXT_COLOUR, 1)
    if status:
        (tw, _), _ = cv2.getTextSize(status, FONT, 0.55, 1)
        cv2.putText(frame, status, (frame.shape[1] - tw - 10, 22), FONT, 0.55, BAR_COLOUR, 1)
    lines = [line for line in lines if line]
    if lines:
        h, row = frame.shape[0], 22
        cv2.rectangle(frame, (0, h - row * len(lines) - 6), (frame.shape[1], h), (0, 0, 0), -1)
        for i, line in enumerate(reversed(lines)):
            cv2.putText(frame, line[:110], (8, h - 8 - i * row), FONT, 0.5, TEXT_COLOUR, 1)
