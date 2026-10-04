import { jsPDF } from "jspdf";
import type { DebriefReport } from "./api";

const MARGIN = 18;
const PAGE_W = 210; // A4 mm
const PAGE_H = 297;
const CONTENT_W = PAGE_W - MARGIN * 2;

const CYAN = [0, 150, 170] as const;
const INK = [30, 34, 44] as const;
const MUTED = [110, 118, 132] as const;
const ROSE = [180, 40, 60] as const;
const GREEN = [20, 130, 90] as const;

export function downloadDebriefPdf(report: DebriefReport) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  let y = MARGIN;

  function ensureRoom(need: number) {
    if (y + need > PAGE_H - MARGIN) {
      doc.addPage();
      y = MARGIN;
    }
  }

  function heading(text: string, size = 14, color: readonly [number, number, number] = INK) {
    ensureRoom(10);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(size);
    doc.setTextColor(...color);
    doc.text(text, MARGIN, y);
    y += size * 0.5;
  }

  function rule() {
    ensureRoom(4);
    doc.setDrawColor(220, 224, 230);
    doc.setLineWidth(0.3);
    doc.line(MARGIN, y, PAGE_W - MARGIN, y);
    y += 5;
  }

  function paragraph(text: string, size = 10, color: readonly [number, number, number] = INK, indent = 0) {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(size);
    doc.setTextColor(...color);
    const lines = doc.splitTextToSize(text, CONTENT_W - indent);
    ensureRoom(lines.length * (size * 0.42) + 2);
    doc.text(lines, MARGIN + indent, y);
    y += lines.length * (size * 0.42) + 2;
  }

  // ---- Header ----
  doc.setFillColor(...CYAN);
  doc.rect(0, 0, PAGE_W, 28, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(255, 255, 255);
  doc.text("DEEP SPACE TRIALS — DEBRIEF REPORT", MARGIN, 17);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.text(`Exam ${report.exam_id}  ·  Generated ${new Date().toLocaleString()}`, MARGIN, 23);
  y = 38;

  // ---- Score summary ----
  const sectorNames = report.sectors.map((s) => s.name);
  paragraph(
    sectorNames.length
      ? `Completed ${report.items_total} trial${report.items_total === 1 ? "" : "s"} across ${sectorNames.join(", ")}.`
      : "No trials were attempted in this session.",
    10, MUTED,
  );
  y += 1;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(28);
  doc.setTextColor(...CYAN);
  doc.text(`${report.score_pct}%`, MARGIN, y + 10);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(...MUTED);
  doc.text(`${report.items_passed} / ${report.items_total} trials cleared`, MARGIN + 32, y + 10);
  y += 16;
  rule();

  // ---- Sector performance ----
  heading("Sector Performance");
  y += 2;
  for (const sec of report.sectors) {
    ensureRoom(8);
    const passed = sec.passed > 0;
    const statusColor = passed ? GREEN : ROSE;
    doc.setFillColor(statusColor[0], statusColor[1], statusColor[2]);
    doc.circle(MARGIN + 1.5, y - 1.5, 1.5, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10.5);
    doc.setTextColor(...INK);
    doc.text(sec.name, MARGIN + 6, y);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(statusColor[0], statusColor[1], statusColor[2]);
    doc.text(passed ? "PASSED" : "NEEDS REVIEW", PAGE_W - MARGIN - 32, y, { align: "left" });
    doc.setTextColor(...MUTED);
    doc.text(`Rating ${sec.rating_after}`, PAGE_W - MARGIN, y, { align: "right" });
    y += 7;
  }
  y += 3;
  rule();

  // ---- Misconceptions ----
  heading("Misconceptions Identified", 14, report.findings.length ? ROSE : GREEN);
  y += 2;
  if (report.findings.length === 0) {
    paragraph("No active misconceptions detected in the trial(s) attempted. Solid understanding shown.", 10, GREEN);
  } else {
    for (const f of report.findings) {
      ensureRoom(14);
      doc.setFillColor(250, 235, 236);
      doc.roundedRect(MARGIN, y - 4, CONTENT_W, 6, 1, 1, "F");
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...ROSE);
      doc.text(`${f.class} · ${f.name}`, MARGIN + 2, y);
      doc.setFont("helvetica", "normal");
      doc.setTextColor(...MUTED);
      doc.text(`On: ${f.trial_title}`, PAGE_W - MARGIN - 2, y, { align: "right" });
      y += 6;
      paragraph(f.subtitle, 9.5, INK, 2);
      for (const ev of f.evidence || []) {
        paragraph(`• [${ev.type}] ${ev.text}`, 8.5, MUTED, 4);
      }
      y += 2;
    }
  }
  rule();

  // ---- Recommendations ----
  heading("Recommended Next Steps");
  y += 2;
  for (const rec of report.recommendations) {
    ensureRoom(12);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(10);
    doc.setTextColor(...CYAN);
    doc.text(`→ ${rec.title}`, MARGIN, y);
    y += 5;
    paragraph(rec.description, 9, MUTED, 4);
    y += 1;
  }

  // ---- Footer page numbers ----
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...MUTED);
    doc.text(`Re:Learn · Black Hole Deep Space Trials — page ${i} of ${pageCount}`, PAGE_W / 2, PAGE_H - 10, { align: "center" });
  }

  doc.save(`debrief-${report.exam_id}.pdf`);
}
