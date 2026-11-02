"""PDF forensic report (ReportLab). Everything shown comes from the stored AnalysisResult and its artifacts."""
from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from .config import DECISION_THRESHOLD, VERSION
from .schemas import AnalysisResult, Classification

NAVY = colors.HexColor("#14213d")
GREY = colors.HexColor("#5b6472")
VERDICT_COLORS = {Classification.REAL: colors.HexColor("#1f8a4c"), Classification.FAKE: colors.HexColor("#c0392b"),
                  Classification.INCONCLUSIVE: colors.HexColor("#b9770e")}

LIMITATIONS = [
    "This report is produced by an automated model and is investigative support, not definitive proof of manipulation or authenticity.",
    "Scores are uncalibrated neural-network outputs; they are not probabilities of the media being fake in the real world.",
    "The model analyses faces only. It has been validated only on the data and generators described in the model evaluation report; "
    "unseen generators, heavy compression, low resolution and adversarial edits can cause false positives and false negatives.",
    "Grad-CAM heat-maps show regions that influenced the RGB stream's fake score. They do not prove that those pixels were altered "
    "and do not explain the noise-residual stream.",
    "A REAL result does not prove authenticity; a FAKE result does not identify who or what created the media.",
]


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1", parent=ss["Title"], textColor=NAVY, fontSize=20, leading=24, alignment=0, spaceAfter=2))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=4))
    ss.add(ParagraphStyle("Small", parent=ss["BodyText"], fontSize=8.5, leading=11, textColor=GREY))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=13))
    return ss


def _kv(rows, widths=(45 * mm, 125 * mm)):
    t = Table([[Paragraph(f"<b>{escape(str(k))}</b>", _S["Small"]), Paragraph(escape(str(v)), _S["Body"])] for k, v in rows], colWidths=widths)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d9dde3")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _img(path: Path, max_w: float, max_h: float) -> Optional[RLImage]:
    if not path.is_file():
        return None
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    s = min(max_w / w, max_h / h, 1.0 if max(w, h) < 4000 else 1.0)
    return RLImage(str(path), width=w * s, height=h * s)


def _timeline(result: AnalysisResult, width=170 * mm, height=48 * mm) -> Optional[Drawing]:
    v = result.video
    if not v or not v.frames:
        return None
    d = Drawing(width, height)
    pad_l, pad_b, pad_t, pad_r = 28, 16, 8, 8
    W, H = width - pad_l - pad_r, height - pad_b - pad_t
    tmax = max(v.duration_s, v.frames[-1].timestamp_s, 1e-6)
    d.add(Rect(pad_l, pad_b, W, H, strokeColor=colors.HexColor("#c8cdd5"), fillColor=colors.HexColor("#f7f8fa")))
    ythr = pad_b + H * DECISION_THRESHOLD
    d.add(Rect(pad_l, ythr, W, H * (1 - DECISION_THRESHOLD), strokeColor=None, fillColor=colors.Color(0.75, 0.22, 0.17, alpha=0.07)))
    d.add(Line(pad_l, ythr, pad_l + W, ythr, strokeColor=colors.HexColor("#c0392b"), strokeDashArray=[3, 2], strokeWidth=0.8))
    pts = []
    for f in v.frames:
        pts += [pad_l + W * f.timestamp_s / tmax, pad_b + H * (f.frame_fake_probability or 0.0)]
    if len(pts) >= 4:
        d.add(PolyLine(pts, strokeColor=NAVY, strokeWidth=1.2))
    for i in range(0, len(pts), 2):
        d.add(Rect(pts[i] - 1.5, pts[i + 1] - 1.5, 3, 3, fillColor=NAVY, strokeColor=None))
    for val in (0.0, 0.5, 1.0):
        d.add(String(pad_l - 4, pad_b + H * val - 3, f"{val:.1f}", fontSize=7, textAnchor="end", fillColor=GREY))
    d.add(String(pad_l, 4, "0 s", fontSize=7, fillColor=GREY))
    d.add(String(pad_l + W, 4, f"{tmax:.1f} s", fontSize=7, textAnchor="end", fillColor=GREY))
    d.add(String(pad_l + 3, pad_b + H - 9, "frame fake-score (red line = 0.5 decision boundary)", fontSize=7, fillColor=GREY))
    return d


_S = _styles()


def _footer(analysis_id: str):
    def draw(c, doc):
        c.saveState()
        c.setFont("Helvetica", 7.5); c.setFillColor(GREY)
        c.drawString(18 * mm, 10 * mm, f"Verif-AI v{VERSION}  |  Analysis {analysis_id}  |  Model-based evidence, not proof")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        c.restoreState()
    return draw


def build_pdf(result: AnalysisResult, artifacts_dir: Optional[Path]) -> bytes:
    art = Path(artifacts_dir) if artifacts_dir else None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"Verif-AI Forensic Report {result.analysis_id}", author="Verif-AI")
    el: List = [Paragraph("Verif-AI Forensic Analysis Report", _S["H1"]),
                Paragraph("See Beyond the Pixels. Verify What's Real.", _S["Small"]), Spacer(1, 6)]

    cls = result.classification or Classification.INCONCLUSIVE
    banner = Table([[Paragraph(f"<font color='white'><b>{cls.value}</b></font>", ParagraphStyle("b", fontSize=18, leading=22)),
                     Paragraph("<font color='white'>"
                               + (f"Fake-score {result.fake_probability:.3f} &nbsp;|&nbsp; Real-score {result.real_probability:.3f}"
                                  if result.fake_probability is not None else "No score could be produced")
                               + "</font>", ParagraphStyle("b2", fontSize=10, leading=14))]], colWidths=[45 * mm, 125 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), VERDICT_COLORS[cls]), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    el += [banner, Spacer(1, 4), Paragraph(escape(result.score_semantics), _S["Small"])]
    if result.decisiveness is not None:
        el.append(Paragraph(f"Distance from the 0.5 decision boundary: {result.decisiveness:.2f} ({result.decisiveness_label}). "
                            "This is not a probability of the verdict being correct.", _S["Small"]))

    m = result.model
    el += [Paragraph("Analysis details", _S["H2"]), _kv([
        ("Analysis ID", result.analysis_id), ("Media type", result.media_type), ("File name", result.filename),
        ("File size", f"{result.file_size_bytes:,} bytes"), ("File SHA-256", result.file_sha256 or "not computed"),
        ("Analysed (UTC)", (result.completed_at or result.created_at).strftime("%Y-%m-%d %H:%M:%S")),
        ("Processing time", f"{result.processing_time_s:.2f} s" if result.processing_time_s is not None else "n/a"),
        ("Resolution", f"{result.image_width} x {result.image_height}" if result.image_width else "n/a"),
        ("Faces analysed", str(result.faces_detected)),
        ("Model", f"{m.name} v{m.version} ({m.architecture})" if m else "n/a"),
        ("Model role", f"{m.registry_label or m.name} [{m.registry_id}] - {m.role_label}" if (m and m.role_label) else "not recorded"),
        ("Model status", m.registry_note if (m and m.registry_note) else "not recorded"),
        ("Checkpoint", f"{m.checkpoint_file}  sha256 {m.checkpoint_sha256[:16]}..." if m else "n/a"),
        ("Device / threshold", f"{m.device} / {m.decision_threshold}" if m else "n/a"),
        ("Aggregation", result.aggregation_note or "n/a")])]

    if result.media_type == "image":
        a = art / result.annotated_artifact if (art and result.annotated_artifact) else None
        im = _img(a, 170 * mm, 110 * mm) if a else None
        if im:
            el += [Paragraph("Detected faces", _S["H2"]), im]
        if result.faces:
            rows = [["#", "Label", "Fake-score", "Real-score", "Detector conf.", "Box (x1,y1,x2,y2)"]]
            for f in result.faces:
                rows.append([str(f.index + 1), f.label.value, f"{f.fake_probability:.4f}", f"{f.real_probability:.4f}", f"{f.detection_confidence:.3f}", ", ".join(f"{v:.0f}" for v in f.box)])
            t = Table(rows, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8cdd5"))]))
            el += [Spacer(1, 6), t]
            pics = []
            for f in result.faces[:6]:
                c = _img(art / f.crop_artifact, 52 * mm, 52 * mm) if (art and f.crop_artifact) else None
                h = _img(art / f.heatmap_artifact, 52 * mm, 52 * mm) if (art and f.heatmap_artifact) else None
                if c and h:
                    pics.append(KeepTogether([Paragraph(f"Face #{f.index + 1}: model input (left) and Grad-CAM (right)", _S["Small"]), Table([[c, h]])]))
            if pics:
                el.append(Paragraph("Grad-CAM evidence", _S["H2"])); el += pics
    else:
        v = result.video
        if v:
            el += [Paragraph("Video analysis", _S["H2"]), _kv([
                ("Duration / FPS", f"{v.duration_s:.1f} s / {v.fps:.2f}"), ("Resolution", f"{v.width} x {v.height}"),
                ("Frames sampled", f"{v.frames_sampled} (at ~{v.sampling_fps} fps, capped)"), ("Frames with a face", str(v.frames_with_faces)),
                ("Corrupt frames skipped", str(v.corrupt_frames_skipped)), ("Face tracks", str(len(v.tracks))),
                ("Aggregation", f"{v.aggregation_method} (not validated on labelled videos)")])]
            tl = _timeline(result)
            if tl:
                el += [Paragraph("Frame-level timeline", _S["H2"]), tl]
            if v.temporal:
                el.append(Paragraph(f"Frame scores: std {v.temporal.score_std:.3f}, mean frame-to-frame change {v.temporal.mean_abs_frame_delta:.3f}, "
                                    f"{v.temporal.suspicious_fraction * 100:.0f}% of frames above 0.5, longest consecutive run {v.temporal.longest_suspicious_run}, "
                                    f"threshold crossings {v.temporal.sign_flips}.", _S["Small"]))
            el.append(Paragraph("Suspicious timestamps", _S["H2"]))
            if v.suspicious_segments:
                rows = [["Start (s)", "End (s)", "Frames", "Max fake-score"]] + [[f"{s.start_s:.2f}", f"{s.end_s:.2f}", str(s.frames), f"{s.max_fake_probability:.3f}"] for s in v.suspicious_segments]
                t = Table(rows, repeatRows=1)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                                       ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8cdd5"))]))
                el.append(t)
            else:
                el.append(Paragraph("No sampled frame scored above the 0.5 decision boundary.", _S["Body"]))
            if v.tracks:
                rows = [["Track", "Frames", "From (s)", "To (s)", "Median score", "Frames > 0.5"]] + [[str(t.track_id), str(t.frames), f"{t.first_timestamp_s:.1f}", f"{t.last_timestamp_s:.1f}", f"{t.median_fake_probability:.3f}", f"{t.suspicious_fraction * 100:.0f}%"] for t in v.tracks[:8]]
                t = Table(rows, repeatRows=1)
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                                       ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8cdd5"))]))
                el += [Spacer(1, 6), t]
            ev = []
            for e in v.evidence_frames[:6]:
                fr = _img(art / e.frame_artifact, 80 * mm, 55 * mm) if art else None
                hm = _img(art / e.heatmap_artifact, 52 * mm, 52 * mm) if (art and e.heatmap_artifact) else None
                cells = [x for x in (fr, hm) if x]
                if cells:
                    ev.append(KeepTogether([Paragraph(f"t = {e.timestamp_s:.2f}s (frame {e.frame_index}), fake-score {e.fake_probability:.3f}", _S["Small"]), Table([cells])]))
            if ev:
                el.append(Paragraph("Highest-scoring frames (evidence)", _S["H2"])); el += ev

    el.append(Paragraph("Explanation", _S["H2"]))
    el += [Paragraph("&bull; " + escape(x), _S["Body"]) for x in result.explanation]
    if result.warnings:
        el.append(Paragraph("Warnings", _S["H2"]))
        el += [Paragraph("&bull; " + escape(x), _S["Body"]) for x in result.warnings]
    el += [Paragraph("Scope", _S["H2"]), Paragraph(escape(result.scope_note), _S["Body"]),
           Paragraph("Limitations and interpretation", _S["H2"])] + [Paragraph("&bull; " + escape(x), _S["Body"]) for x in LIMITATIONS]
    doc.build(el, onFirstPage=_footer(result.analysis_id), onLaterPages=_footer(result.analysis_id))
    return buf.getvalue()
