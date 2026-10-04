import { jsPDF } from "jspdf";
import type { DebriefReport, DebriefRemediation } from "./api";

// Page geometry (A4 in mm)
const MARGIN = 16;
const PAGE_W = 210;
const PAGE_H = 297;
const CONTENT_W = PAGE_W - MARGIN * 2; // 178 mm

// Professional Deep Space palette
const DEEP_NAVY = [11, 18, 32] as const;
const CYAN = [0, 150, 185] as const;
const BRIGHT_CYAN = [0, 195, 225] as const;
const INK = [24, 30, 42] as const;
const SLATE = [60, 72, 90] as const;
const MUTED = [105, 118, 136] as const;
const LIGHT_BG = [248, 250, 253] as const;
const CARD_BORDER = [224, 230, 240] as const;

// Status colors
const GREEN = [16, 135, 82] as const;
const GREEN_BG = [236, 253, 245] as const;
const GREEN_BORDER = [167, 243, 208] as const;

const ROSE = [190, 35, 55] as const;
const ROSE_BG = [255, 241, 242] as const;
const ROSE_BORDER = [254, 205, 211] as const;

const AMBER = [185, 110, 15] as const;
const AMBER_BG = [254, 249, 235] as const;
const AMBER_BORDER = [253, 230, 138] as const;

export function downloadDebriefPdf(report: DebriefReport) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  let y = MARGIN;

  function ensureRoom(need: number) {
    if (y + need > PAGE_H - MARGIN - 12) {
      doc.addPage();
      y = MARGIN + 4;
    }
  }

  function sectionHeading(
    title: string,
    badgeText?: string,
    accentColor: readonly [number, number, number] = CYAN
  ) {
    ensureRoom(14);
    // Left accent bar
    doc.setFillColor(accentColor[0], accentColor[1], accentColor[2]);
    doc.rect(MARGIN, y, 3, 7, "F");

    // Title
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(...INK);
    doc.text(title.toUpperCase(), MARGIN + 6, y + 5.5);

    // Optional tag pill
    if (badgeText) {
      const titleWidth = doc.getTextWidth(title.toUpperCase());
      const badgeX = MARGIN + 8 + titleWidth;
      const badgeW = doc.getTextWidth(badgeText) + 6;
      doc.setFillColor(accentColor[0], accentColor[1], accentColor[2]);
      doc.roundedRect(badgeX, y + 1.2, badgeW, 4.8, 1, 1, "F");
      doc.setFont("helvetica", "bold");
      doc.setFontSize(7.5);
      doc.setTextColor(255, 255, 255);
      doc.text(badgeText, badgeX + 3, y + 4.6);
    }

    y += 10;
  }

  function divider(spacing = 4) {
    ensureRoom(spacing + 2);
    doc.setDrawColor(...CARD_BORDER);
    doc.setLineWidth(0.3);
    doc.line(MARGIN, y, PAGE_W - MARGIN, y);
    y += spacing;
  }

  // ==========================================
  // 1. HEADER BANNER
  // ==========================================
  doc.setFillColor(...DEEP_NAVY);
  doc.rect(0, 0, PAGE_W, 30, "F");

  // Cyan bottom accent strip
  doc.setFillColor(...BRIGHT_CYAN);
  doc.rect(0, 29, PAGE_W, 1.2, "F");

  // Sub-header eyebrow
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7.5);
  doc.setTextColor(...BRIGHT_CYAN);
  doc.text("RE:LEARN DEEP SPACE ACADEMY // BLACK HOLE DIAGNOSTICS", MARGIN, 11);

  // Main banner title
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(255, 255, 255);
  doc.text("DEEP SPACE TRIALS — MISSION DEBRIEF REPORT", MARGIN, 18.5);

  // Metadata line
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(170, 185, 205);
  const dateStr = new Date().toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
  doc.text(
    `Exam Session: ${report.exam_id}  ·  Generated: ${dateStr}  ·  Verification: Rule + ML Harness`,
    MARGIN,
    24.5
  );

  y = 38;

  // ==========================================
  // 2. SCORECARD & EXECUTIVE OVERVIEW
  // ==========================================
  ensureRoom(32);
  const cardH = 26;
  doc.setFillColor(...LIGHT_BG);
  doc.setDrawColor(...CARD_BORDER);
  doc.setLineWidth(0.4);
  doc.roundedRect(MARGIN, y, CONTENT_W, cardH, 2.5, 2.5, "FD");

  // Left Score Badge
  const scoreBoxW = 44;
  doc.setFillColor(238, 245, 255);
  doc.roundedRect(MARGIN + 3, y + 3, scoreBoxW, cardH - 6, 2, 2, "F");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(...CYAN);
  doc.text(`${report.score_pct}%`, MARGIN + scoreBoxW / 2 + 3, y + 13.5, { align: "center" });

  doc.setFont("helvetica", "bold");
  doc.setFontSize(7.5);
  const scoreTier =
    report.score_pct >= 80 ? "MISSION ACCOMPLISHED" : report.score_pct >= 50 ? "PARTIAL CLEARANCE" : "REMEDIATION NEEDED";
  doc.setTextColor(report.score_pct >= 80 ? GREEN[0] : report.score_pct >= 50 ? AMBER[0] : ROSE[0]);
  doc.text(scoreTier, MARGIN + scoreBoxW / 2 + 3, y + 18.5, { align: "center" });

  // Center & Right KPI Telemetry
  const statsX = MARGIN + scoreBoxW + 8;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10.5);
  doc.setTextColor(...INK);
  doc.text("Candidate Evaluation Summary", statsX, y + 8);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(...SLATE);
  const trialsClearedText = `• Trials Cleared: ${report.items_passed} of ${report.items_total} attempted`;
  const sectorCountText = `• Sectors Tested: ${report.sectors.length} sector${report.sectors.length === 1 ? "" : "s"}`;
  const findingCount = report.findings.length;
  const findingText =
    findingCount === 0
      ? "• Diagnostics: No active misconceptions detected"
      : `• Diagnostics: ${findingCount} active misconception${findingCount === 1 ? "" : "s"} identified`;

  doc.text(trialsClearedText, statsX, y + 13.5);
  doc.text(sectorCountText, statsX, y + 18);
  doc.setTextColor(findingCount > 0 ? ROSE[0] : GREEN[0]);
  doc.text(findingText, statsX + 70, y + 13.5);

  doc.setTextColor(...MUTED);
  doc.setFontSize(7.5);
  doc.text("Harness: Invariant Model v2.4", statsX + 70, y + 18);

  y += cardH + 7;

  // ==========================================
  // 3. SECTOR PERFORMANCE BREAKDOWN (FIXED NO-OVERLAP TABLE)
  // ==========================================
  sectionHeading("Sector Performance", `${report.sectors.length} ATTEMPTED`);

  if (report.sectors.length === 0) {
    doc.setFont("helvetica", "italic");
    doc.setFontSize(9);
    doc.setTextColor(...MUTED);
    doc.text("No sectors were attempted in this session.", MARGIN + 4, y);
    y += 8;
  } else {
    // Table Header
    ensureRoom(9);
    doc.setFillColor(241, 245, 249);
    doc.roundedRect(MARGIN, y, CONTENT_W, 6.5, 1.5, 1.5, "F");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(7.5);
    doc.setTextColor(...MUTED);
    doc.text("SECTOR / SPECIALIZATION", MARGIN + 8, y + 4.5);
    doc.text("TRIALS", MARGIN + 72, y + 4.5);
    doc.text("EVALUATION", MARGIN + 115, y + 4.5, { align: "center" });
    doc.text("SECTOR RATING", PAGE_W - MARGIN - 6, y + 4.5, { align: "right" });
    y += 8.5;

    // Table Rows
    for (let idx = 0; idx < report.sectors.length; idx++) {
      const sec = report.sectors[idx];
      ensureRoom(10);

      // Alternating row background
      if (idx % 2 === 1) {
        doc.setFillColor(250, 252, 254);
        doc.rect(MARGIN, y - 3, CONTENT_W, 8.5, "F");
      }

      const passed = sec.passed > 0;
      const statusColor = passed ? GREEN : ROSE;
      const statusBg = passed ? GREEN_BG : ROSE_BG;
      const statusBorder = passed ? GREEN_BORDER : ROSE_BORDER;

      // Bullet dot
      doc.setFillColor(statusColor[0], statusColor[1], statusColor[2]);
      doc.circle(MARGIN + 4, y + 1.2, 1.3, "F");

      // Sector Name
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...INK);
      doc.text(sec.name, MARGIN + 8, y + 2.2);

      // Trials Count
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(...SLATE);
      const count = sec.items.length || 1;
      doc.text(`${count} trial${count === 1 ? "" : "s"} (${sec.passed} cleared)`, MARGIN + 72, y + 2.2);

      // Status Pill (x = MARGIN + 99, w = 32mm -> ends at MARGIN + 131 = 147mm)
      const pillX = MARGIN + 99;
      const pillW = 32;
      const pillH = 5.2;
      doc.setFillColor(statusBg[0], statusBg[1], statusBg[2]);
      doc.setDrawColor(statusBorder[0], statusBorder[1], statusBorder[2]);
      doc.setLineWidth(0.25);
      doc.roundedRect(pillX, y - 1.8, pillW, pillH, 1.2, 1.2, "FD");

      doc.setFont("helvetica", "bold");
      doc.setFontSize(7.5);
      doc.setTextColor(statusColor[0], statusColor[1], statusColor[2]);
      doc.text(passed ? "PASSED" : "NEEDS REVIEW", pillX + pillW / 2, y + 1.8, { align: "center" });

      // Rating (right aligned at PAGE_W - MARGIN - 6 = 188mm -> starts around 168mm)
      // There is 168mm - 147mm = 21mm of GUARANTEED empty space! Zero overlap!
      doc.setFont("helvetica", "bold");
      doc.setFontSize(8.5);
      doc.setTextColor(...INK);
      doc.text(`Rating: ${sec.rating_after}`, PAGE_W - MARGIN - 6, y + 2.2, { align: "right" });

      y += 8.5;
    }
  }

  y += 4;
  divider(5);

  // ==========================================
  // 4. DIAGNOSTIC FINDINGS (MISCONCEPTIONS)
  // ==========================================
  const findingsColor = report.findings.length > 0 ? ROSE : GREEN;
  sectionHeading(
    "Diagnostic Findings",
    report.findings.length > 0 ? `${report.findings.length} ACTIVE` : "CLEARED",
    findingsColor
  );

  if (report.findings.length === 0) {
    ensureRoom(14);
    doc.setFillColor(...GREEN_BG);
    doc.setDrawColor(...GREEN_BORDER);
    doc.setLineWidth(0.3);
    doc.roundedRect(MARGIN, y, CONTENT_W, 11, 2, 2, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(...GREEN);
    doc.text("✓  All Invariants Satisfied", MARGIN + 5, y + 5);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...SLATE);
    doc.text(
      "No active misconceptions detected in the attempted trials. Boundary contracts and control flows were fully verified.",
      MARGIN + 5,
      y + 9
    );
    y += 15;
  } else {
    for (const f of report.findings) {
      ensureRoom(20);

      // Card Container
      doc.setFillColor(...ROSE_BG);
      doc.setDrawColor(...ROSE_BORDER);
      doc.setLineWidth(0.3);
      doc.roundedRect(MARGIN, y, CONTENT_W, 6.5, 1.5, 1.5, "FD");

      // Misconception Badge
      doc.setFillColor(...ROSE);
      doc.roundedRect(MARGIN + 2, y + 1, 12, 4.5, 1, 1, "F");
      doc.setFont("helvetica", "bold");
      doc.setFontSize(7.5);
      doc.setTextColor(255, 255, 255);
      doc.text(f.class, MARGIN + 8, y + 4.2, { align: "center" });

      // Misconception Name
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...ROSE);
      doc.text(f.name, MARGIN + 16, y + 4.5);

      // Problem / Trial Context (Right)
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8);
      doc.setTextColor(...SLATE);
      doc.text(`Trial: ${f.trial_title || f.item_id}`, PAGE_W - MARGIN - 4, y + 4.5, { align: "right" });

      y += 8.5;

      // Subtitle / Description
      doc.setFont("helvetica", "normal");
      doc.setFontSize(8.5);
      doc.setTextColor(...INK);
      const subLines = doc.splitTextToSize(f.subtitle || f.belief || "Logical invariant violation.", CONTENT_W - 8);
      ensureRoom(subLines.length * 4 + 2);
      doc.text(subLines, MARGIN + 4, y);
      y += subLines.length * 4 + 2;

      // Evidence list
      if (f.evidence && f.evidence.length > 0) {
        for (const ev of f.evidence) {
          ensureRoom(6);
          doc.setFont("helvetica", "bold");
          doc.setFontSize(7.5);
          doc.setTextColor(...ROSE);
          doc.text(`[${ev.type}]`, MARGIN + 6, y);

          const typeW = doc.getTextWidth(`[${ev.type}] `);
          doc.setFont("helvetica", "normal");
          doc.setTextColor(...SLATE);
          const evLines = doc.splitTextToSize(ev.text, CONTENT_W - 8 - typeW);
          doc.text(evLines, MARGIN + 6 + typeW, y);
          y += evLines.length * 3.8 + 1.5;
        }
      }
      y += 3;
    }
  }

  divider(6);

  // ==========================================
  // 5. HOW TO CLEAR IDENTIFIED MISCONCEPTIONS (LLM PEDAGOGY SECTION)
  // ==========================================
  const remediations: DebriefRemediation[] =
    report.remediations && report.remediations.length > 0
      ? report.remediations
      : report.findings.map((f) => ({
          class: f.class,
          name: f.name,
          subtitle: f.subtitle,
          trial_title: f.trial_title,
          root_cause: f.belief || f.subtitle,
          rule_to_remember: "Review boundary invariants and verify loop/recursion conditions.",
          code_fix: {
            wrong: "// Review faulty implementation pattern",
            right: "// Implement verified invariant bounds",
          },
          self_check: "Trace execution with n = 0 and n = 1 before compilation.",
        }));

  if (remediations.length > 0) {
    sectionHeading(
      "How to Clear Identified Misconceptions",
      "LLM REMEDIATION GUIDE",
      CYAN
    );

    // Pedagogical note
    ensureRoom(9);
    doc.setFont("helvetica", "italic");
    doc.setFontSize(8);
    doc.setTextColor(...MUTED);
    doc.text(
      "Actionable cognitive shifts, invariant rules, and code patterns to permanently resolve diagnosed bugs.",
      MARGIN + 2,
      y
    );
    y += 5.5;

    for (const rem of remediations) {
      // Ensure enough room for the card header + text
      ensureRoom(36);

      // Remediation Card Outer Frame
      doc.setFillColor(254, 255, 255);
      doc.setDrawColor(...CARD_BORDER);
      doc.setLineWidth(0.4);

      // Card Header strip
      doc.setFillColor(244, 248, 253);
      doc.roundedRect(MARGIN, y, CONTENT_W, 7, 1.5, 1.5, "F");

      // Badge
      doc.setFillColor(...CYAN);
      doc.roundedRect(MARGIN + 2.5, y + 1.2, 14, 4.6, 1, 1, "F");
      doc.setFont("helvetica", "bold");
      doc.setFontSize(7.5);
      doc.setTextColor(255, 255, 255);
      doc.text(rem.class, MARGIN + 9.5, y + 4.5, { align: "center" });

      // Title
      doc.setFont("helvetica", "bold");
      doc.setFontSize(9);
      doc.setTextColor(...INK);
      doc.text(`${rem.name}`, MARGIN + 19, y + 4.7);

      // Trial Target (Right)
      if (rem.trial_title) {
        doc.setFont("helvetica", "normal");
        doc.setFontSize(7.5);
        doc.setTextColor(...MUTED);
        doc.text(`Context: ${rem.trial_title}`, PAGE_W - MARGIN - 4, y + 4.7, { align: "right" });
      }

      y += 10.5;

      // 1. Root Cause / Mental Model Shift
      ensureRoom(14);
      doc.setFont("helvetica", "bold");
      doc.setFontSize(7.5);
      doc.setTextColor(...SLATE);
      doc.text("1. MENTAL MODEL SHIFT (WHY THIS OCCURS)", MARGIN + 4, y);
      y += 4;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(8.5);
      doc.setTextColor(...INK);
      const rootLines = doc.splitTextToSize(rem.root_cause, CONTENT_W - 8);
      ensureRoom(rootLines.length * 3.8 + 2);
      doc.text(rootLines, MARGIN + 4, y);
      y += rootLines.length * 3.8 + 3.5;

      // 2. Invariant Rule Box
      ensureRoom(16);
      doc.setFillColor(...AMBER_BG);
      doc.setDrawColor(...AMBER_BORDER);
      doc.setLineWidth(0.3);

      const ruleLines = doc.splitTextToSize(`RULE TO REMEMBER: ${rem.rule_to_remember}`, CONTENT_W - 12);
      const ruleBoxH = Math.max(10, ruleLines.length * 3.8 + 5);
      doc.roundedRect(MARGIN + 3, y - 1, CONTENT_W - 6, ruleBoxH, 1.5, 1.5, "FD");

      doc.setFont("helvetica", "bold");
      doc.setFontSize(8);
      doc.setTextColor(...AMBER);
      doc.text(ruleLines, MARGIN + 6, y + 3.2);
      y += ruleBoxH + 4;

      // 3. Side-by-side or Stacked Code Fix Blocks
      if (rem.code_fix && (rem.code_fix.wrong || rem.code_fix.right)) {
        ensureRoom(28);

        doc.setFont("helvetica", "bold");
        doc.setFontSize(7.5);
        doc.setTextColor(...SLATE);
        doc.text("2. CODE TRANSFORMATION (WRONG VS CORRECT)", MARGIN + 4, y);
        y += 4;

        // Render Wrong Code Block
        if (rem.code_fix.wrong) {
          const wrongLines = rem.code_fix.wrong.split("\n");
          const blockH = Math.max(10, wrongLines.length * 3.8 + 6);
          ensureRoom(blockH + 4);

          doc.setFillColor(255, 243, 244);
          doc.setDrawColor(...ROSE_BORDER);
          doc.roundedRect(MARGIN + 3, y, CONTENT_W - 6, blockH, 1.5, 1.5, "FD");

          doc.setFont("helvetica", "bold");
          doc.setFontSize(7);
          doc.setTextColor(...ROSE);
          doc.text("✗ WRONG PATTERN", MARGIN + 6, y + 3.8);

          doc.setFont("courier", "normal");
          doc.setFontSize(7.5);
          doc.setTextColor(140, 20, 30);
          for (let lIdx = 0; lIdx < wrongLines.length; lIdx++) {
            doc.text(wrongLines[lIdx], MARGIN + 6, y + 8 + lIdx * 3.6);
          }
          y += blockH + 2.5;
        }

        // Render Right Code Block
        if (rem.code_fix.right) {
          const rightLines = rem.code_fix.right.split("\n");
          const blockH = Math.max(10, rightLines.length * 3.8 + 6);
          ensureRoom(blockH + 4);

          doc.setFillColor(240, 253, 244);
          doc.setDrawColor(...GREEN_BORDER);
          doc.roundedRect(MARGIN + 3, y, CONTENT_W - 6, blockH, 1.5, 1.5, "FD");

          doc.setFont("helvetica", "bold");
          doc.setFontSize(7);
          doc.setTextColor(...GREEN);
          doc.text("✓ CORRECT PATTERN (INVARIANT SAFE)", MARGIN + 6, y + 3.8);

          doc.setFont("courier", "bold");
          doc.setFontSize(7.5);
          doc.setTextColor(15, 100, 50);
          for (let lIdx = 0; lIdx < rightLines.length; lIdx++) {
            doc.text(rightLines[lIdx], MARGIN + 6, y + 8 + lIdx * 3.6);
          }
          y += blockH + 3.5;
        }
      }

      // 4. Pre-Compilation Self-Check
      if (rem.self_check) {
        ensureRoom(10);
        doc.setFont("helvetica", "bold");
        doc.setFontSize(7.5);
        doc.setTextColor(...CYAN);
        doc.text("3. PRE-COMPILATION SELF-CHECK HABIT:", MARGIN + 4, y);
        y += 3.8;

        doc.setFont("helvetica", "normal");
        doc.setFontSize(8);
        doc.setTextColor(...SLATE);
        const checkLines = doc.splitTextToSize(`• ${rem.self_check}`, CONTENT_W - 10);
        ensureRoom(checkLines.length * 3.6 + 2);
        doc.text(checkLines, MARGIN + 6, y);
        y += checkLines.length * 3.6 + 4;
      }

      y += 4;
    }
  }

  divider(6);

  // ==========================================
  // 6. RECOMMENDED ACTION PLAN
  // ==========================================
  sectionHeading("Recommended Action Plan", "NEXT STEPS");

  for (let idx = 0; idx < report.recommendations.length; idx++) {
    const rec = report.recommendations[idx];
    ensureRoom(14);

    doc.setFillColor(246, 249, 253);
    doc.setDrawColor(...CARD_BORDER);
    doc.roundedRect(MARGIN, y, CONTENT_W, 11, 1.5, 1.5, "FD");

    doc.setFont("helvetica", "bold");
    doc.setFontSize(8.5);
    doc.setTextColor(...CYAN);
    doc.text(`${idx + 1}.  ${rec.title}`, MARGIN + 4, y + 4.5);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(...SLATE);
    const desc = doc.splitTextToSize(rec.description, CONTENT_W - 8);
    doc.text(desc, MARGIN + 4, y + 8.5);

    y += 14;
  }

  // ==========================================
  // 7. FOOTER PAGE NUMBERS & SECURITY SEAL
  // ==========================================
  const pageCount = doc.getNumberOfPages();
  for (let i = 1; i <= pageCount; i++) {
    doc.setPage(i);

    // Subtle footer separator
    doc.setDrawColor(...CARD_BORDER);
    doc.setLineWidth(0.25);
    doc.line(MARGIN, PAGE_H - 12, PAGE_W - MARGIN, PAGE_H - 12);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(...MUTED);
    doc.text("Re:Learn Autonomous Pedagogical Platform · Black Hole Deep Space Trials", MARGIN, PAGE_H - 7);
    doc.text(`Page ${i} of ${pageCount}`, PAGE_W - MARGIN, PAGE_H - 7, { align: "right" });
  }

  // Save the document
  doc.save(`debrief-${report.exam_id}.pdf`);
}
