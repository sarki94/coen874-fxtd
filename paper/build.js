// Builds the IEEE-style two-column A4 paper (.docx) from content.js + tables.json.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, SectionType, BorderStyle, TabStopType, Footer, PageNumber, VerticalAlign,
} = require("docx");
const C = require("./content.js");
const TB = JSON.parse(fs.readFileSync(__dirname + "/tables.json", "utf8"));

// ---------- geometry (IEEE A4 conference template) ----------
const MM = 56.6929; // DXA per mm
const PAGE = { width: 11906, height: 16838 };
const MARGIN = { top: Math.round(19 * MM), bottom: Math.round(43 * MM), left: Math.round(14.32 * MM), right: Math.round(14.32 * MM) };
const TEXTW = PAGE.width - MARGIN.left - MARGIN.right;           // ~10282
const GAP = Math.round(4.22 * MM);                                 // ~239
const COLW = Math.floor((TEXTW - GAP) / 2);                        // ~5021
const FONT = "Times New Roman";
const BODY = 20; // half-points = 10 pt

// ---------- citations ----------
const order = [];
function citeNum(key) {
  if (!C.refs[key]) throw new Error("unknown ref " + key);
  let i = order.indexOf(key);
  if (i < 0) { order.push(key); i = order.length - 1; }
  return i + 1;
}
function citeText(keys) {
  const nums = keys.map(citeNum).sort((a, b) => a - b);
  const parts = []; let i = 0;
  while (i < nums.length) {
    let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    if (j - i >= 2) parts.push(`[${nums[i]}]–[${nums[j]}]`);
    else for (let k = i; k <= j; k++) parts.push(`[${nums[k]}]`);
    i = j + 1;
  }
  return parts.join(", ");
}

// ---------- inline markup -> runs ----------
function runs(text, base = {}) {
  text = text.replace(/\{cite:([^}]+)\}/g, (_, k) => citeText(k.split(",").map(s => s.trim())));
  const out = [];
  let i = 0, buf = "", st = { bold: false, italics: false, code: false };
  const flush = (extra = {}) => {
    if (!buf) return;
    out.push(new TextRun({
      text: buf, font: st.code ? "Courier New" : (base.font || FONT), size: st.code ? (base.size || BODY) - 2 : (base.size || BODY),
      bold: base.bold || st.bold, italics: (base.italics || st.italics) !== (extra.invertItalic || false) ? true : false,
      smallCaps: base.smallCaps, color: base.color, ...extra.script,
    }));
    buf = "";
  };
  while (i < text.length) {
    if (text.startsWith("**", i)) { flush(); st.bold = !st.bold; i += 2; continue; }
    if (text[i] === "*" ) { flush(); st.italics = !st.italics; i += 1; continue; }
    if (text[i] === "`") { flush(); st.code = !st.code; i += 1; continue; }
    if ((text.startsWith("_{", i) || text.startsWith("^{", i))) {
      flush();
      const sub = text[i] === "_";
      const end = text.indexOf("}", i);
      const inner = text.slice(i + 2, end);
      buf = inner;
      flush({ script: sub ? { subScript: true } : { superScript: true } });
      i = end + 1; continue;
    }
    buf += text[i]; i++;
  }
  flush();
  return out;
}

// ---------- block builders ----------
let h1n = 0, h2n = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
const para = (text, opts = {}) => new Paragraph({
  alignment: opts.align ?? AlignmentType.JUSTIFIED,
  indent: opts.noIndent ? undefined : { firstLine: 202 },
  spacing: { before: opts.before ?? 0, after: opts.after ?? 0, line: 240 },
  keepNext: opts.keepNext, keepLines: opts.keepLines,
  children: runs(text, opts.base || {}),
});
function h1(text, numbered = true) {
  h2n = 0;
  const label = numbered ? `${ROMAN[h1n++]}. ${text}` : text;
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
    children: [new TextRun({ text: label, font: FONT, size: BODY, smallCaps: true })],
  });
}
function h2(text) {
  const label = `${String.fromCharCode(65 + h2n++)}. ${text}`;
  return new Paragraph({
    alignment: AlignmentType.LEFT, spacing: { before: 120, after: 60 }, keepNext: true,
    children: runs("*" + label + "*"),
  });
}
function eq(text, n, width) {
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(width / 2) }, { type: TabStopType.RIGHT, position: width }],
    spacing: { before: 80, after: 80 }, keepLines: true,
    children: [new TextRun({ text: "\t", font: FONT, size: BODY }), ...runs(text, { italics: true }),
      new TextRun({ text: `\t(${n})`, font: FONT, size: BODY })],
  });
}
let figN = 0, tabN = 0;
function figure(b) {
  figN++;
  const png = fs.readFileSync(b.file);
  const w = png.readUInt32BE(16), h = png.readUInt32BE(20);
  const wpx = Math.round(b.widthIn * 96), hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: png, transformation: { width: wpx, height: hpx },
        altText: { title: `Fig. ${figN}`, description: b.caption.replace(/[*_^{}]/g, ""), name: b.fig } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 0, after: 160 },
      children: [new TextRun({ text: `Fig. ${figN}.  `, font: FONT, size: 16 }), ...runs(b.caption, { size: 16 })] }),
  ];
}
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(caption, header, rows, widths, opts = {}) {
  tabN++;
  const size = opts.size || 15;
  const total = widths.reduce((a, b) => a + b, 0);
  const mkCell = (txt, w, { head = false, first = false, last = false, align } = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    borders: { top: first ? thin : none, bottom: head || last ? thin : none, left: none, right: none },
    children: [new Paragraph({ alignment: align ?? (opts.alignLeftCols?.includes(txt.__col) ? AlignmentType.LEFT : AlignmentType.CENTER),
      spacing: { before: 0, after: 0 }, children: runs(String(txt), { size, bold: head }) })],
  });
  const headRows = (Array.isArray(header[0]) ? header : [header]).map((hr, ri, all) => new TableRow({
    tableHeader: true, cantSplit: true,
    children: hr.map((t, ci) => {
      const span = typeof t === "object" ? t.span : 1;
      const text = typeof t === "object" ? t.text : t;
      const w = widths.slice(ci, ci + span).reduce((a, b) => a + b, 0);
      return new TableCell({
        columnSpan: span, width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 20, bottom: 20, left: 50, right: 50 },
        borders: { top: ri === 0 ? thin : none, bottom: ri === all.length - 1 || (typeof t === "object" && t.rule) ? thin : none, left: none, right: none },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 0 }, children: runs(text, { size, bold: true }) })],
      });
    }),
  }));
  const bodyRows = rows.map((r, ri) => new TableRow({ cantSplit: true,
    children: r.map((t, ci) => mkCell(t, widths[ci], { last: ri === rows.length - 1,
      align: (opts.leftCols || []).includes(ci) ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 0 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${ROMAN[tabN - 1]}`, font: FONT, size: 16, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 60 }, keepNext: true,
      children: runs(caption, { size: 16, smallCaps: true }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      rows: [...headRows, ...bodyRows] }),
    ...(opts.note ? [new Paragraph({ spacing: { before: 40, after: 160 }, alignment: AlignmentType.JUSTIFIED,
      children: runs(opts.note, { size: 14 }) })] : [new Paragraph({ spacing: { after: 120 }, children: [] })]),
  ];
}
const pretty = s => s.replace(/V\*\(s0\)/g, "V^{*}(s_{0})").replace(/\bs0\b/g, "s_{0}").replace(/gamma/g, "γ")
  .replace(/8\*row/g, "8·row").replace(/Delta = 1\/\(2\^b - 1\)/, "Δ = 1/(2^{b} − 1)").replace(/ x ε/g, " × ε")
  .replace(/(\d)-(\d)/g, "$1–$2").replace(/P\(s'\|s,a\)/, "P(s′|s,a)").replace(/R\(s,a,s'\)/, "R(s,a,s′)")
  .replace(/Environment ID/, "Environment ID").replace(/is_slippery=True/, "is_slippery=True");

function tableBlock(name) {
  if (name === "mdp") return { wide: false, items: table("Environment and MDP specification",
    ["Element", "Specification"], TB.mdp.map(r => [pretty(r[0]), pretty(r[1])]), [1550, COLW - 1550],
    { leftCols: [0, 1], size: 15 }) };
  if (name === "hyper") return { wide: false, items: table("Training, tuning and evaluation settings",
    ["Setting", "Value"], TB.hyper.map(r => [pretty(r[0]), pretty(r[1])]), [1650, COLW - 1650],
    { leftCols: [0, 1], size: 15 }) };
  if (name === "main") {
    const w = [760, 620, 1420, 760, 800, 760, 1420, 760, 800, 760, 760];
    const scale = TEXTW / w.reduce((a, b) => a + b, 0);
    const ws = w.map(x => Math.floor(x * scale));
    return { wide: true, items: table("Main results (20 seeds per configuration)",
      [[{ text: "", span: 2 }, { text: "Q-learning", span: 4, rule: true }, { text: "SARSA", span: 4, rule: true }, { text: "", span: 1 }],
       ["Prec.", "Round.", "Retained [95% CI]", "Success", "Steps to 80%", "Not stored", "Retained [95% CI]", "Success", "Steps to 80%", "Not stored", "Bytes"]],
      TB.main, ws, { size: 15,
        note: "Retained: median *V*^{π}(*s*_{0})/*V*^{*}(*s*_{0}) with 95% bootstrap CI. Success: mean greedy success rate over 1000 rollouts (optimal policy: 0.664). Steps to 80%: median environment steps until *V*^{π}(*s*_{0}) ≥ 0.8*V*^{*}(*s*_{0}). Not stored: fraction of non-zero TD updates that left the stored value unchanged. Bytes: bit-packed size of the 64 × 4 table." }) };
  }
  if (name === "stats") {
    const w = [430, 900, 330, 900, 1040, 600, 820];
    const scale = COLW / w.reduce((a, b) => a + b, 0);
    return { wide: false, items: table("Hypothesis tests on final performance",
      ["Hyp.", "Algorithm", "*b*", "Contrast", "Medians", "δ", "*p*_{Holm}"], TB.stats, w.map(x => Math.floor(x * scale)),
      { size: 14, note: "Medians are retained *V*^{π}(*s*_{0})/*V*^{*}(*s*_{0}) for the two groups in the order of the contrast (H1: SR / RTN; H2: Q-learning / SARSA). H1: one-sided Mann–Whitney *U*; H2: two-sided. *p* is Holm-corrected within each family; δ is Cliff’s delta. 20 seeds per group." }) };
  }
  if (name === "bias") {
    const w = [700, 640, (COLW - 1340) / 2, (COLW - 1340) / 2].map(Math.floor);
    return { wide: false, items: table("Value bias: mean of max_{a} Q(s,a) − V^{*}(s) over non-terminal states",
      ["Prec.", "Round.", "Q-learning [95% CI]", "SARSA [95% CI]"], TB.bias, w, { size: 14,
        note: "Positive values mean the learned table overestimates the optimal values. SARSA learns the value of its ε-greedy policy, so its values sit below *V*^{*} by design." }) };
  }
}

// ---------- assemble sections ----------
const sections = [];
let current = [];
let mode = "single"; // title block first
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] });
const pageProps = { size: PAGE, margin: { ...MARGIN, header: 400, footer: 1200 } };
function closeSection(cols, type) {
  sections.push({ properties: { type, page: pageProps, column: cols === 2 ? { count: 2, space: GAP, equalWidth: true } : { count: 1 } },
    footers: { default: footer }, children: current });
  current = [];
}

// Title block (single column)
current.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 },
  children: [new TextRun({ text: C.meta.title, font: FONT, size: 44 })] }));
current.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
  children: [new TextRun({ text: C.meta.author, font: FONT, size: 22 })] }));
C.meta.affil.forEach((a, i) => current.push(new Paragraph({ alignment: AlignmentType.CENTER,
  spacing: { after: i === C.meta.affil.length - 1 ? 280 : 0 }, children: [new TextRun({ text: a, font: FONT, size: 18, italics: true })] })));
let firstSection = true;
function startTwoCol() { closeSection(1, firstSection ? undefined : SectionType.CONTINUOUS); firstSection = false; }
startTwoCol();

// Abstract + index terms
current.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 },
  children: [new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "—", font: FONT, size: 18, bold: true }), ...runs(C.meta.abstract, { size: 18, bold: true })] }));
current.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
  children: [new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }),
    new TextRun({ text: "—" + C.meta.keywords, font: FONT, size: 18, bold: true })] }));

function emitBlocks(blocks) {
  for (const b of blocks) {
    if (b.h1) current.push(h1(b.h1));
    else if (b.h1nonum) current.push(h1(b.h1nonum, false));
    else if (b.h2) current.push(h2(b.h2));
    else if (b.p) current.push(para(b.p, b.align === "left" ? { align: AlignmentType.LEFT } : {}));
    else if (b.eq) current.push(eq(b.eq, b.n, COLW));
    else if (b.fig || b.table) {
      const tb = b.fig ? null : tableBlock(b.table);
      const wide = b.fig ? b.wide : tb.wide;
      const items = b.fig ? figure(b) : tb.items;
      if (wide) {
        closeSection(2, SectionType.CONTINUOUS);      // end the 2-column run so far
        current.push(...items);
        closeSection(1, SectionType.CONTINUOUS);      // full-width block
      } else current.push(...items);
    }
  }
}
emitBlocks(C.body);

// References
current.push(h1("References", false));
order.length; // citations already numbered by order of appearance
order.forEach((k, i) => current.push(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 }, indent: { left: 400, hanging: 400 },
  tabStops: [{ type: TabStopType.LEFT, position: 400 }],
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 16 }), ...runs(C.refs[k], { size: 16 })] })));
const unused = Object.keys(C.refs).filter(k => !order.includes(k));
if (unused.length) console.warn("UNCITED refs:", unused.join(", "));

// Appendix (after references, excluded from the page limit)
emitBlocks(C.appendix);
closeSection(2, SectionType.CONTINUOUS);

const doc = new Document({
  creator: C.meta.author, title: C.meta.title, description: "COEN874 term project paper",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections,
});
Packer.toBuffer(doc).then(buf => {
  const out = process.argv[2] || "/home/claude/paper/out/COEN874_Iliyasu_FixedPoint_TD.docx";
  fs.mkdirSync(require("path").dirname(out), { recursive: true });
  fs.writeFileSync(out, buf);
  console.log("wrote", out, "| refs cited:", order.length, "| figures:", figN, "| tables:", tabN);
});
