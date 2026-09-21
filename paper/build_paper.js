/* Builds the research paper from data/processed/paper_data.json and figures/*.png.
   Run:  node paper/build_paper.js   ->  paper/Buyer_Segmentation_Research_Paper.docx          */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType,
  HeadingLevel, LevelFormat, BorderStyle, Header, Footer, PageNumber, PageBreak, VerticalAlign,
} = require('docx');

const ROOT = path.resolve(__dirname, '..');
const D = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/processed/paper_data.json'), 'utf8'));
const R = D.res, De = D.desc, G = D.geo, IM = D.images, MS = D.model_selection, HY = D.hypotheses, INV = D.inv_vs_personal, V = D.versions;
const SEGS = ['S1', 'S2', 'S3', 'S4'];
const P = D.profile, PO = D.profile_overall, NAME = Object.fromEntries(SEGS.map(s => [s, D.meta[s].name]));

/* ---------- formatting helpers ---------- */
const n0 = x => Math.round(x).toLocaleString('en-US');
const p0 = x => (x * 100).toFixed(0) + '%';
const p1 = x => (x * 100).toFixed(1) + '%';
const mn = s => s.replace(/^-/, '\u2212');
const d1 = x => mn(x.toFixed(1)), d2 = x => mn(x.toFixed(2)), d3 = x => mn(x.toFixed(3));
const pm = x => (x < 0 ? '\u2212' : '') + Math.abs(x * 100).toFixed(0) + '%';
const usd = x => '$' + n0(x);
const usdK = x => '$' + Math.round(x / 1e3) + 'k';
const usdM = x => '$' + (x / 1e6).toFixed(2) + 'M';
const pv = p => (p < 0.001 ? '< 0.001' : p.toFixed(3));
const rngOf = (field, fmt, keys = SEGS) => { const v = keys.map(s => P[s][field]); return fmt(Math.min(...v)) + '–' + fmt(Math.max(...v)); };
const msk = k => MS.find(r => r.k === k);

/* ---------- palette / layout ---------- */
const NAVY = '1F4E79', ORANGE = 'E07B39', GREYTXT = '5B6B7A', LINE = 'D0D7DE', BAND = 'F5F8FB', HILITE = 'FDF1E6';
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440, CONTENT = PAGE_W - 2 * MARGIN; // 9026 DXA
const FONT = 'Calibri';

/* ---------- inline markup: **bold**  //italic// ---------- */
function runs(text, base = {}) {
  return String(text).split(/(\*\*[^*]+\*\*|\/\/[^/]+\/\/)/).filter(s => s !== '').map(s => {
    if (s.startsWith('**')) return new TextRun({ text: s.slice(2, -2), bold: true, ...base });
    if (s.startsWith('//')) return new TextRun({ text: s.slice(2, -2), italics: true, ...base });
    return new TextRun({ text: s, ...base });
  });
}
const P_ = (text, o = {}) => new Paragraph({ children: runs(text, o.run || {}), alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, before: o.before ?? 0 }, keepNext: o.keepNext, indent: o.indent });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const BUL = (t, o = {}) => new Paragraph({ numbering: { reference: 'bullets', level: 0 }, children: runs(t, o.run || {}), spacing: { after: o.after ?? 70 }, alignment: AlignmentType.LEFT });
const NUM = (ref, t) => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 80 }, alignment: AlignmentType.LEFT });
const CODE = t => new Paragraph({ children: [new TextRun({ text: t, font: 'Courier New', size: 17 })], shading: { type: ShadingType.CLEAR, fill: 'F3F6F9', color: 'auto' }, spacing: { after: 0 }, indent: { left: 200 } });

/* ---------- numbering of figures / tables (by order of appearance) ---------- */
const FIGNUM = { client_profile: 1, purchase_behaviour: 2, market_context: 3, association: 4, model_selection: 5, dendrogram: 6, pca_importance: 7, profile_heatmap: 8, composition: 9, geography: 10, time_and_k5: 11, hypotheses: 12 };
const FILE = { client_profile: 'fig01_client_profile', purchase_behaviour: 'fig02_purchase_behaviour', market_context: 'fig03_market_context', association: 'fig04_association', model_selection: 'fig05_model_selection', dendrogram: 'fig06_dendrogram', pca_importance: 'fig07_pca_importance', profile_heatmap: 'fig08_profile_heatmap', composition: 'fig09_composition', geography: 'fig10_geography', time_and_k5: 'fig12_time_and_k5', hypotheses: 'fig11_hypotheses' };
const F = k => 'Figure ' + FIGNUM[k];
let lastFig = 0;
function fig(key, widthIn, caption, alt) {
  const n = FIGNUM[key]; if (n !== lastFig + 1) throw new Error(`figure order: ${key} is #${n} but previous was #${lastFig}`); lastFig = n;
  const im = IM[FILE[key]]; const w = Math.round(widthIn * 96), h = Math.round(w * im.h / im.w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: 'png', data: fs.readFileSync(im.path), transformation: { width: w, height: h }, altText: { title: `Figure ${n}`, description: alt, name: FILE[key] } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 }, children: [new TextRun({ text: `Figure ${n}. `, bold: true, size: 18 }), ...runs(caption, { size: 18 })] }),
  ];
}
const TABNUM = { sources: '1', cleaning: '2', features: '3', selection: '4', profile: '5', invper: '6', personas: '7', playbook: '8', A1: 'A1', B1: 'B1', C1: 'C1' };
const T = k => 'Table ' + TABNUM[k];
const tcap = (k, text) => new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: `Table ${TABNUM[k]}. `, bold: true, size: 18 }), ...runs(text, { size: 18 })] });
const tnote = text => new Paragraph({ spacing: { before: 60, after: 200 }, alignment: AlignmentType.JUSTIFIED, children: runs(text, { size: 16, color: GREYTXT }) });

/* ---------- tables ---------- */
const bd = { style: BorderStyle.SINGLE, size: 4, color: LINE }, nob = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
function cell(content, width, o = {}) {
  const size = o.size || 17; const kn = !!o.keepNext; const align = o.align === 'right' ? AlignmentType.RIGHT : o.align === 'center' ? AlignmentType.CENTER : AlignmentType.LEFT;
  const items = Array.isArray(content) ? content : [content];
  const kids = items.map((it, idx) => {
    if (it && typeof it === 'object' && it.bullet) return new Paragraph({ keepNext: kn, numbering: { reference: 'bullets', level: 0 }, children: runs(it.bullet, { size }), spacing: { after: 40 }, alignment: AlignmentType.LEFT });
    const isSub = it && typeof it === 'object' && it.sub;
    const txt = it && typeof it === 'object' ? (it.sub || it.t) : it;
    return new Paragraph({ keepNext: kn, alignment: align, spacing: { after: 0 }, children: runs(String(txt ?? ''), { size: isSub ? size - 2 : size, bold: !!o.bold && !isSub, color: o.color || (isSub ? 'D6E2EE' : undefined) }) });
  });
  return new TableCell({ width: { size: width, type: WidthType.DXA }, verticalAlign: o.top ? VerticalAlign.TOP : VerticalAlign.CENTER, margins: { top: 50, bottom: 50, left: 80, right: 80 },
    shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: 'auto' } : undefined, borders: { top: bd, bottom: bd, left: nob, right: nob }, children: kids });
}
function table(headers, rows, widths, o = {}) {
  if (widths.reduce((a, b) => a + b, 0) !== CONTENT) throw new Error('column widths must sum to ' + CONTENT + ' got ' + widths.reduce((a, b) => a + b, 0));
  const al = o.align || headers.map(() => 'left');
  const head = new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, widths[i], { fill: NAVY, color: 'FFFFFF', bold: true, align: al[i] === 'right' ? 'center' : al[i], size: o.size, keepNext: true })) });
  const body = rows.map((r, ri) => {
    const row = Array.isArray(r) ? { cells: r } : r;
    const fill = row.fill || (ri % 2 ? BAND : undefined);
    return new TableRow({ cantSplit: true, children: row.cells.map((c, i) => cell(c, widths[i], { fill, align: al[i], size: o.size, bold: row.bold || (o.boldFirst && i === 0), top: o.top, keepNext: o.keep !== false && ri < rows.length - 1 })) });
  });
  return new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}

/* =====================================================================================
   CONTENT
   ===================================================================================== */
const body = [];
const add = (...x) => x.flat().forEach(e => body.push(e));

/* ---------- title block + abstract ---------- */
add(
  new Paragraph({ spacing: { before: 200, after: 80 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: 'RESEARCH PAPER', bold: true, size: 18, color: ORANGE, characterSpacing: 40 })] }),
  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: 'Machine Learning based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence', bold: true, size: 44, color: NAVY })] }),
  new Paragraph({ spacing: { after: 160 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: `A clustering study of ${n0(R.n_clients)} clients and ${n0(R.log.raw_properties)} property listings`, size: 26, color: GREYTXT })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: NAVY, space: 6 } }, spacing: { after: 160 }, children: [] }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.LEFT, children: runs('**Prepared for:** Parcl Co. Limited and Unified Mentor', { size: 21 }) }),
  new Paragraph({ spacing: { after: 20 }, alignment: AlignmentType.LEFT, children: runs('**Author:** [Author name]', { size: 21 }) }),
  new Paragraph({ spacing: { after: 240 }, alignment: AlignmentType.LEFT, children: runs('**Date:** 21 September 2026', { size: 21 }) }),
  new Paragraph({ spacing: { before: 60, after: 80 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: 'Abstract', bold: true, size: 26, color: NAVY })] }),
  P_(`Parcl Co. Limited sells apartments and office units in twenty towers and wants to know who its buyers are. We combined its client register (${n0(R.n_clients)} clients) with its sales ledger (${n0(R.log.raw_properties)} listings, ${n0(R.sold_units)} sold) to build a profile of every client, grouped the clients with K-Means clustering, and checked the result against Ward hierarchical clustering. A rule fixed in advance, which ranks candidate cluster counts on five criteria, selects four segments: ${NAME.S1} (${p0(P.S1.client_share)} of clients), ${NAME.S2} (${p0(P.S2.client_share)}), ${NAME.S3} (${p0(P.S3.client_share)}) and ${NAME.S4} (${p0(P.S4.client_share)}). The segments are reproducible (bootstrap agreement ${d2(R.stability)}; agreement with Ward clustering ${d2(R.ward_ari)}) but only weakly separated (mean silhouette ${d2(R.silhouette)}). The reason is that client attributes in this dataset are almost statistically independent of one another: the median pairwise association is ${d2(R.assoc_median)}. A decision tree with three questions reproduces every client's segment, so the segments differ in //how// clients buy (country of residence, investment or personal use, loan or cash) and not in age, satisfaction, units bought or spend. We also tested the four buyer personas suggested in the project brief and found no support for any of them. We recommend using the segments to shape messaging and financing offers rather than to rank clients by value, and we list the additional data needed to identify richer personas. An interactive Streamlit dashboard accompanies the paper.`),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.LEFT, children: runs('**Keywords:** customer segmentation; K-Means; hierarchical clustering; cluster validation; real estate; investment profiling', { size: 19 }) }),
  new Paragraph({ children: [new PageBreak()] }),
);

/* ---------- 1. Introduction ---------- */
add(H1('1. Introduction'),
  P_('Parcl Co. Limited (Parcl) sells apartments and office units in twenty towers to a client base spread over ten countries. Knowing which kinds of buyer it serves, and how they differ, is the starting point for targeted marketing, financing partnerships and inventory planning. The project brief, prepared with Unified Mentor, asks for a machine-learning segmentation of Parcl\'s buyers and an investment profile of each segment, delivered as a research paper and a live dashboard.'),
  P_('The brief also proposes four buyer personas as a starting hypothesis: Global Investors (C1), First-Time Buyers (C2), Corporate Buyers (C3) and Luxury Investors (C4). We treat these as hypotheses to be tested, not as answers the clustering has to reproduce. That choice shapes the paper. Where the data support a claim we say so; where they do not, we say that too.'),
  P_('The study has five objectives:', { after: 60, keepNext: true }),
  NUM('n_obj', 'Clean the client register and the sales ledger and combine them into one client-level table.'),
  NUM('n_obj', 'Segment clients with K-Means and hierarchical clustering, choose the number of segments by an objective rule, and measure how much confidence the result deserves.'),
  NUM('n_obj', 'Profile each segment on investment behaviour, geography and demographics.'),
  NUM('n_obj', 'Test the four personas from the brief against the data.'),
  NUM('n_obj', 'Turn the findings into recommendations, and expose the analysis through an interactive dashboard.'),
  P_(`Section 2 describes the data and how we cleaned it. Section 3 explores the client base, purchase behaviour and market context. Section 4 sets out the method, Section 5 the results, and Section 6 the tests of the four personas. Section 7 gives insights and recommendations, Section 8 the limitations, and Section 9 concludes.`, { before: 60 }),
);

/* ---------- 2. Data ---------- */
add(H1('2. Data'), H2('2.1 Sources'),
  tcap('sources', 'The two source files.'),
  table(['File', 'Rows', 'Contents'], [
    ['clients.csv', n0(R.log.raw_clients), 'One row per client: type (individual or company), name, date of birth, gender, country, region, acquisition purpose, satisfaction score (1–5), loan applied, referral channel.'],
    ['properties.csv', n0(R.log.raw_properties), 'One row per listing: tower, unit category (apartment or office), unit number, floor area, sale price, listing status, transaction month, and a client reference on sold units.'],
  ], [1500, 900, 6626], { boldFirst: true, size: 18, align: ['left', 'right', 'left'] }),
  P_(`Of the ${n0(R.log.raw_properties)} listings, ${n0(R.log.sold_listings)} are sold and ${n0(R.log.available_listings)} are available. Every sold listing carries a client reference, and all ${n0(R.log.sold_listings)} match a client in the register. Each of the ${n0(R.n_clients)} clients bought at least three units (maximum ${De.units_max}), and ${p0(De.share_3_4)} bought three or four. Transaction dates run from ${R.log.date_min.slice(0, 7)} to ${R.log.date_max.slice(0, 7)} and always fall on the first of a month, so they identify the month of a sale rather than the day.`, { before: 120 }),
  P_('The client register holds no income, net worth or prior-ownership field, and it does not record purchase behaviour. This matters because three of the four personas in the brief are defined by income, first-time-buyer status or investment size. We therefore derive behaviour from the sales ledger (Section 2.4) and treat income and first-time status as unobserved.'),
);

add(H2('2.2 Data quality and cleaning'),
  P_('The files are unusually clean. Table 2 lists every issue we looked for, what we found, and what we did.', { keepNext: true }),
  tcap('cleaning', 'Data-quality checks and the action taken for each.'),
  table(['Issue', 'Finding', 'Action'], [
    ['Missing values', `None in either file (${R.log.missing_client_cells} missing client cells).`, 'Imputation rules (median for numbers, "Unknown" for categories) kept in the code so that reruns on new data are safe.'],
    ['Duplicates', `${R.log.dup_client_ids} duplicate client IDs, ${R.log.dup_client_rows_ignoring_id} duplicate client rows, ${R.log.dup_listing_ids} duplicate listing IDs.`, 'De-duplication step retained; nothing removed.'],
    ['Referential integrity', `${R.log.orphan_client_refs} sold listings point to an unknown client; ${R.log.sold_without_client} sold listings have no client.`, 'None needed.'],
    ['Category labels', 'Consistent spelling and case. The source calls corporate clients "Company" and personal-use purchases "Home".', 'Renamed to "Corporate" and "Personal Use" to match the brief; "M"/"F" expanded.'],
    ['Sale price', 'Stored as text with "$" and thousands separators.', 'Converted to numbers.'],
    ['Birth dates', `Two formats mixed: ${n0(R.log.dob_slash_format)} written month/day/year, ${n0(R.log.dob_dash_format)} with hyphens and both parts at 12 or below.`, 'Parsed by format; the hyphen convention was inferred (Section 2.3) and tested for sensitivity.'],
    ['Corporate clients', 'Companies carry a gender and a date of birth, presumably those of a contact person.', 'Kept as given; flagged as a caveat.'],
    ['Unit numbers', 'Not unique within a tower: 1,119 distinct tower–unit pairs across 10,000 listings.', '//listing_id// used as the record key; no claim is made about resales.'],
    ['Personal identifiers', 'Names and dates of birth are not needed once age is computed.', 'Dropped from all processed files.'],
  ], [1700, 3700, 3626], { boldFirst: true, size: 17, top: true }),
);

add(H2('2.3 The date-of-birth convention'),
  P_(`Birth dates arrive in two formats. The ${n0(R.log.dob_slash_format)} dates written with slashes are month/day/year and unambiguous, because the day part reaches 31. The ${n0(R.log.dob_dash_format)} written with hyphens have both parts at 12 or below, so 05-11-1968 could be 5 November or 11 May. We settled the convention with //properties.csv//, which uses the same hyphenated format for every one of its 10,000 transaction dates. Read as month-day-year, those dates are the first day of each of 24 consecutive months, January 2024 to December 2025. Read as day-month-year, they would all fall in January of 2024 or 2025. The first reading is clearly the intended one, so we read hyphenated birth dates as month-day-year too. This is an inference from a sibling file, not a documented fact, so we checked that it does not matter: recomputing every age under the alternative reading changes it by ${d2(R.log.dob_ambiguity_mean_abs_age_diff)} years on average, and by ${d2(R.log.dob_ambiguity_max_abs_age_diff)} years at most.`),
);

add(H2('2.4 Client-level features'),
  P_('Because only sold listings carry a client reference, purchase behaviour is measured from the sold units of each client. For every client we compute the number of units, total spend, average unit price, average price per square foot, average floor area, the share of purchases that are office units, the number of towers bought in, and the months of first and last purchase. Age is measured at 31 December 2025, the end of the observation window. Clients with five or more units are flagged as portfolio buyers, which marks the heavy tail of the unit-count distribution (Section 3.2).'),
);

/* ---------- 3. EDA ---------- */
add(H1('3. Exploratory analysis'), H2('3.1 The client base'),
  P_(`The client base is ${p0(De.usa_share)} US-resident (${n0(De.country_counts.USA)} clients), and California alone accounts for ${n0(De.california)} of them (${p0(De.california / R.n_clients)}). The other nine countries contribute between ${De.country_counts.Denmark} (Denmark) and ${De.country_counts.UK} (United Kingdom) clients each. There are ${De.n_corp} corporate clients (${p0(De.n_corp / R.n_clients)}). Ages run from ${d1(De.age_min)} to ${d1(De.age_max)} years (mean ${d1(De.age_mean)}, standard deviation ${d1(De.age_sd)}) with no dominant age group. Corporate contacts are younger than individuals (mean ${d1(De.age_corp)} against ${d1(De.age_ind)} years; Mann–Whitney p ${pv(De.corp_age_p)}), and none is older than 66. ${p1(De.inv_share)} of clients bought as an investment and ${p1(De.loan_share)} applied for a loan. The website brought in ${p1(De.ref.Website)} of clients, agencies ${p1(De.ref.Agency)} and other clients ${p1(De.ref.Client)}. Satisfaction scores are spread almost evenly over 1 to 5 (each score is chosen by 19–21% of clients), giving a mean of ${d2(De.sat_mean)}; ${p1(De.sat_low_share)} of clients scored 1 or 2.`),
  ...fig('client_profile', 6.27, 'The client base. (a) Age at end of 2025 by client type. (b) Country of residence. (c) Satisfaction score against a uniform benchmark. (d–f) Acquisition purpose, loan application and referral channel.', 'Six-panel summary of client age, country, satisfaction, purpose, loan and referral channel'),
);

add(H2('3.2 Purchase behaviour'),
  P_(`Every client bought at least three units, ${p0(De.share_3_4)} bought three or four, and the tail reaches ${De.units_max} (${F('purchase_behaviour')}a). We call the ${R.portfolio_clients} clients with five or more units portfolio buyers (${p0(R.portfolio_clients / R.n_clients)} of clients). Total spend per client has a median of ${usdM(De.spend_median)} (mean ${usdM(De.spend_mean)}, range ${usdM(De.spend_min)} to ${usdM(De.spend_max)}), and the average unit price per client is ${usdK(De.avg_price_mean)}. Sold units average ${usd(R.ppsf_mean)} per square foot (standard deviation ${usd(R.ppsf_sd)}); offices sell at a modest premium over apartments (${usd(R.ppsf_office)} against ${usd(R.ppsf_apt)}). Price is almost entirely a function of floor area: the correlation between the two is ${d2(R.area_price_r)}.`),
  ...fig('purchase_behaviour', 6.27, 'Purchase behaviour of sold units. (a) Units bought per client, log scale; orange marks portfolio buyers (five or more). (b) Total spend per client. (c) Price per square foot by unit type.', 'Histograms of units per client, total spend per client and price per square foot'),
);

add(H2('3.3 Market context'),
  P_(`Listings and sales fell sharply at the start of 2025 (${F('market_context')}a). Monthly listings averaged ${n0(R.listings_2024_per_month)} in 2024 and ${n0(R.listings_2025_per_month)} in 2025 (${pm(De.listed_drop)}), and monthly sales averaged ${n0(R.sold_2024_per_month)} and ${n0(R.sold_2025_per_month)} (${pm(De.sold_drop)}). The fall is a step rather than a slide: ${De.dec24_listed} listings in December 2024 and ${De.jan25_listed} in January 2025. Revenue from sold units followed, from $${(De.rev_2024 / 1e9).toFixed(2)} billion in 2024 to $${(De.rev_2025 / 1e9).toFixed(2)} billion in 2025. The share of listings that sold barely moved (${p1(R.sold_2024_per_month / R.listings_2024_per_month)} in 2024, ${p1(R.sold_2025_per_month / R.listings_2025_per_month)} in 2025; ${p0(R.sell_through)} overall), which points to fewer listings rather than weaker demand, although the data cannot say why fewer units were listed.`),
  P_(`Prices were flat. The monthly average price per square foot stays between ${usd(De.ppsf_month_min)} and ${usd(De.ppsf_month_max)} with no trend (Spearman rho = ${d3(R.ppsf_time_rho)}, p = ${d2(R.ppsf_time_p)}). Average listing prices differ between towers, from ${usdK(De.tower_price_min)} to ${usdK(De.tower_price_max)}, but price per square foot hardly differs at all (${usd(De.tower_ppsf_min)} to ${usd(De.tower_ppsf_max)}), so towers differ in unit size and not in pricing.`),
  ...fig('market_context', 6.27, 'Market context. (a) Monthly listings and sold units. (b) Average sale price per square foot by month; the dashed line is the overall mean. (c) Average listing price by tower.', 'Monthly listings and sales, price per square foot over time, and average listing price by tower'),
);

add(H2('3.4 How related are the client attributes?'),
  P_(`Before clustering we measured the association between every pair of client attributes: Cramér\'s V (Cramér, 1946) for two categorical variables, the correlation ratio (η) for a categorical and a numeric variable, and the absolute Spearman correlation for two numeric variables (${F('association')}). The picture is unusually flat. The median association across all 66 pairs is ${d2(R.assoc_median)}. The only large values are arithmetic identities: total spend is built from units and average price (${d2(R.top_assoc_pairs[0][2])} and ${d2(R.top_assoc_pairs[1][2])}). Outside those, the strongest pairs are units bought with average unit price (${d2(R.top_assoc_pairs[2][2])}; larger portfolios lean towards cheaper units) and client type with age (${d2(R.top_assoc_pairs[3][2])}; the corporate contact-age gap noted above). Purpose, loan, referral channel, satisfaction, gender and country group are unrelated to one another and to how much a client buys.`),
  P_('This has a direct consequence for the analysis. A clustering algorithm finds groups only where attributes move together. When they do not, any partition is a way of slicing independent variables, and low separation scores are to be expected. We therefore committed in advance to report separation and stability plainly, and to test the brief\'s personas directly instead of assuming that clusters would confirm them.'),
  ...fig('association', 5.3, 'Pairwise association between client attributes: bias-corrected Cramér\'s V (categorical–categorical), correlation ratio η (categorical–numeric) and absolute Spearman rho (numeric–numeric). 0 means no association and 1 a perfect one.', 'Heatmap of pairwise associations between twelve client attributes, almost all near zero'),
);

add(H2('3.5 Revenue concentration'),
  P_(`Revenue is not concentrated. The top 10% of clients by spend account for ${p1(R.top10_share)} of revenue and the top 20% for ${p1(R.top20_share)}. The ${R.portfolio_clients} portfolio buyers, ${p0(R.portfolio_clients / R.n_clients)} of clients, contribute ${p1(R.portfolio_rev_share)}. There is no small group of very large accounts to protect or to chase.`),
);

/* ---------- 4. Method ---------- */
add(H1('4. Method'), H2('4.1 Features and encoding'),
  P_(`The clustering uses ${R.n_features} model-matrix columns built from five numeric features and six categorical ones (${T('features')}). Categorical variables are one-hot encoded. Fifty-seven regions are too many for distance-based clustering, so we keep the eleven US states with at least 40 clients, pool the remaining US clients as "US – Other" and pool all non-US clients as "International"; country is encoded separately, so no non-US detail is lost. Gender is left out of the clustering: it is balanced (${p0(De.male_share)} male) and unrelated to every other attribute, so it would only add noise. It is kept for profiling. Label (ordinal) codes are also generated for the categorical fields and exported for use in other tools, but they are never used in distance calculations, because integer codes would impose an order on categories that have none.`),
  tcap('features', 'Feature set used for clustering.'),
  table(['Block', 'Variables', 'Treatment'], [
    ['Demographics', 'Age (years at 31 Dec 2025)', 'MinMax scaled to [0, 1]'],
    ['Sentiment', 'Satisfaction score (1–5)', 'MinMax scaled to [0, 1]'],
    ['Purchase behaviour', 'Units bought (log), average unit price (log), share of purchases that are offices', 'Log transform for skew, then MinMax scaled'],
    ['Client and intent', 'Client type, acquisition purpose, loan applied', 'One-hot'],
    ['Channel', 'Referral channel (3 levels)', 'One-hot'],
    ['Geography', 'Country (10 levels), region group (13 levels)', 'One-hot'],
  ], [1900, 4300, 2826], { boldFirst: true, size: 18 }),
);

const dA = D.design[0], dB = D.design[1], dC = D.design[2], dD = D.design[3], dE = D.design[4];
add(H2('4.2 Scaling'),
  P_(`Distance-based clustering needs features on comparable scales. We use MinMax scaling so that every numeric feature lies in [0, 1], the same range as a one-hot indicator. We compared four alternatives (Appendix C). The brief\'s literal recipe, with age and satisfaction as the only numeric features, gives similar separation (silhouette ${d2(dA.silhouette_k4)} against ${d2(dB.silhouette_k4)} at k = 4) but lower bootstrap stability (${d2(dA.stability_k4)} against ${d2(dB.stability_k4)}), and it cannot express the purchase-based personas. Standardising the numeric features but not the indicators weights the numeric block heavily; stability then falls to ${d2(dC.stability_k4)} and the two algorithms barely agree (ARI ${d2(dC.ward_ari_k4)}). Standardising every column, indicators included, inflates rare categories: an indicator for a 15-client country becomes worth more than ten standard deviations, and the k = 4 solution then contains a cluster of only ${dD.smallest_k4} clients with stability ${d2(dD.stability_k4)}. Equal-variance blocks give the 5% corporate indicator similar leverage, and that design\'s smallest cluster has ${dE.smallest_k4} clients, the size of the whole corporate group. The MinMax design is not the one with the highest silhouette; it is the one with the highest stability, and it is the only one that also covers purchase behaviour.`),
);

add(H2('4.3 Algorithms'),
  P_('We fit K-Means (Lloyd\'s algorithm with k-means++ initialisation and 50 restarts for the final model; MacQueen, 1967; Lloyd, 1982) and Ward\'s hierarchical clustering (Ward, 1963), both with Euclidean distance. The brief asks for both. Agreement between two algorithms with different objectives is also useful evidence that a structure is not an artefact of one of them. All random choices use a fixed seed (42).'),
);

add(H2('4.4 Choosing the number of clusters'),
  P_(`We searched k = 2 to 10 but allowed only k = 3 to 8 to win: two groups is not a segmentation, and more than eight is more than a marketing team can act on. For each candidate we compute five measures: the silhouette score (Rousseeuw, 1987), the Davies–Bouldin index (Davies and Bouldin, 1979), the Calinski–Harabasz index (Caliński and Harabasz, 1974), bootstrap stability, and agreement between K-Means and Ward. Bootstrap stability is the mean adjusted Rand index (Hubert and Arabie, 1985) between the full-data solution and K-Means refitted on 30 random 80% subsamples. We rank the candidates on each measure and choose the lowest mean rank, with ties going to the smaller k. We fixed this rule before looking at the results. We also report the elbow, located with a Kneedle-style rule (Satopää et al., 2011).`),
  P_(`One caution applies. With mostly binary features, the silhouette tends to rise with k because clusters begin to coincide with exact combinations of categories (it peaks at k = 10 here). A rising silhouette curve is therefore not evidence of better structure, which is why the rule combines five criteria and caps k at 8.`),
);

add(H2('4.5 Validation and interpretation'),
  BUL('**Stability.** Bootstrap adjusted Rand index, as above.'),
  BUL('**Algorithm agreement.** Adjusted Rand index between K-Means and Ward, and the share of clients assigned to the same segment by both.'),
  BUL('**Cohesion.** The mean silhouette of each segment, to show which segments are tight and which are loose.'),
  BUL('**What defines a segment.** A depth-3 decision tree and a random forest (300 trees) predict segment from thirteen interpretable binary and numeric attributes; permutation importance shows which attributes matter.'),
  BUL('**Do the segments differ?** Kruskal–Wallis tests (with ε² effect sizes) for numeric variables and chi-square tests (with Cramér\'s V) for categorical ones (Appendix B).', { after: 120 }),
);

add(H2('4.6 Testing the personas'),
  P_('For each persona in the brief we state the claim it implies, choose a test, and report an effect size and a bootstrap 95% interval next to the p-value. We treat an effect below 0.10 in magnitude as negligible (Cohen, 1988). Section 6 gives the results.'),
  H2('4.7 Reproducibility'),
  P_(`The analysis is a set of Python scripts run in order (Appendix D) with fixed random seeds. We used Python ${V.python}, scikit-learn ${V.sklearn} (Pedregosa et al., 2011), SciPy ${V.scipy}, pandas ${V.pandas}, NumPy ${V.numpy}, Matplotlib ${V.matplotlib}, Plotly ${V.plotly} and Streamlit ${V.streamlit}. Every number in this paper is read from the pipeline outputs when the document is built.`),
);

/* ---------- 5. Results ---------- */
const m3 = msk(3), m4 = msk(4), m5 = msk(5), m6 = msk(6);
add(H1('5. Results'), H2('5.1 Selecting the number of segments'),
  P_(`${F('model_selection')} and ${T('selection')} show the diagnostics for k = 2 to 10. The silhouette score is low and nearly flat for small k (${d2(m5.silhouette_kmeans)}–${d2(m3.silhouette_kmeans)} for k up to 6) and climbs at larger k, the pattern one expects when clusters start to coincide with exact combinations of categories. No k reaches the 0.26 that Kaufman and Rousseeuw (1990) give as the lower bound of weak structure. Among the candidates from 3 to 8, k = 4 has the best mean rank (${d1(m4.mean_rank)}), ahead of k = 5 (${d1(m5.mean_rank)}) and k = 6 (${d1(m6.mean_rank)}). It also has the highest bootstrap stability of any k from 2 to 10 (${d2(m4.stability_ari)}), and its K-Means solution agrees closely with Ward clustering (ARI ${d2(m4.kmeans_vs_ward_ari)}), where k = 3 agrees much less (ARI ${d2(m3.kmeans_vs_ward_ari)}). The silhouette at k = 3 is a little higher (${d2(m3.silhouette_kmeans)} against ${d2(m4.silhouette_kmeans)}), a gap too small to matter.`),
  ...fig('model_selection', 6.27, 'Model-selection diagnostics for K-Means (and Ward where shown). The shaded band marks the chosen k = 4. (a) Elbow. (b) Silhouette; the grey zone is the range Kaufman and Rousseeuw (1990) describe as no substantial structure. (c) Davies–Bouldin, lower is better. (d) Calinski–Harabasz, higher is better. (e) Bootstrap stability with one standard deviation. (f) Agreement between K-Means and Ward.', 'Six panels of clustering diagnostics against the number of clusters'),
  tcap('selection', 'Diagnostics for every k tried. The chosen k = 4 is highlighted; the mean rank is computed over k = 3 to 8 only.'),
  table(['k', 'Inertia', 'Silhouette K-Means', 'Silhouette Ward', 'Davies–Bouldin', 'Calinski–Harabasz', 'Stability (ARI)', 'K-Means vs Ward (ARI)', 'Smallest cluster', 'Mean rank'],
    MS.map(r => ({ cells: [String(r.k), n0(r.inertia), d3(r.silhouette_kmeans), d3(r.silhouette_ward), d2(r.davies_bouldin), n0(r.calinski_harabasz), d3(r.stability_ari), d3(r.kmeans_vs_ward_ari), String(r.smallest_cluster), r.mean_rank == null ? '–' : d1(r.mean_rank)], fill: r.k === R.best_k ? HILITE : undefined, bold: r.k === R.best_k })),
    [560, 900, 950, 850, 900, 1000, 950, 1066, 950, 900], { size: 16, align: ['center', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right', 'right'] }),
  tnote('Higher is better for silhouette, Calinski–Harabasz, stability and agreement; lower is better for inertia and Davies–Bouldin. Stability is the mean over 30 bootstrap resamples.'),
  P_(`The elbow rule points to k = ${R.elbow_k}, and k = 5 has a slightly better Davies–Bouldin index (${d2(m5.davies_bouldin)} against ${d2(m4.davies_bouldin)}). What the fifth cluster adds is a split of ${NAME.S1} by referral channel alone: ${D.k5.split['4']} clients who came through the website against ${D.k5.split['3']} who came through an agency or another client (${F('time_and_k5')}b). That is a channel distinction and not a behavioural one, and it leaves the other three segments untouched. We therefore keep four segments and note the channel split as an optional refinement.`),
);

add(H2('5.2 Hierarchical validation'),
  P_(`Cutting the Ward tree at four clusters (${F('dendrogram')}) reproduces the K-Means assignment for ${p1(R.ward_agreement_pct)} of clients (adjusted Rand index ${d2(R.ward_ari)}). Agreement is complete for ${NAME.S1} (${p0(P.S1.ward_agreement)}), high for ${NAME.S3} (${p0(P.S3.ward_agreement)}), and lower for ${NAME.S2} (${p0(P.S2.ward_agreement)}) and ${NAME.S4} (${p0(P.S4.ward_agreement)}). Ward\'s own silhouette at k = 4 is ${d2(R.silhouette_ward)}, a little below K-Means (${d2(R.silhouette)}).`),
  ...fig('dendrogram', 6.0, 'Ward dendrogram truncated to the last 14 merges; leaf labels give the number of clients in each branch. The dashed line cuts the tree into four clusters.', 'Ward hierarchical clustering dendrogram cut at four clusters'),
);

add(H2('5.3 What defines a segment'),
  P_(`The first two principal components carry only ${p1(R.pca_var[0])} and ${p1(R.pca_var[1])} of the variance in the ${R.n_features}-column model matrix, so a two-dimensional picture can show only part of the structure. ${F('pca_importance')}a shows what it does show. The points form many small islands, each a distinct combination of categorical attributes, and each segment is a union of islands that interleave with the islands of other segments. Well-separated clusters would appear as distinct clouds; these do not.`),
  P_(`To see what does define a segment, we trained a decision tree limited to three levels to predict each client\'s segment from thirteen interpretable attributes. It reproduces the segment of every client (accuracy ${p0(R.tree_acc)}; five-fold cross-validated accuracy ${p0(R.tree_cv)}) with three questions. Is the client resident outside the United States? Then the client is in ${NAME.S3}. Otherwise, is the purchase an investment? Then ${NAME.S2}. Otherwise, was a loan applied for? Then ${NAME.S4}; if not, ${NAME.S1}. A random forest tells the same story (${F('pca_importance')}b): shuffling US residence, investment purpose or loan status costs ${d2(D.importance[2].importance)}–${d2(D.importance[0].importance)} in accuracy, and shuffling any of the other ten attributes, including age, satisfaction, units bought, average price and referral channel, costs nothing.`),
  ...fig('pca_importance', 6.27, 'What defines a segment. (a) Clients projected on the first two principal components (small random jitter added so that overlapping points are visible). (b) Permutation importance of the seven most important attributes in a random forest that predicts segment.', 'Principal component scatter of clients coloured by segment, and a bar chart of permutation importance'),
  P_('This is the central result of the paper. The four segments are real in the sense that they are stable and reproducible, but they are a four-way partition of three categorical variables that the K-Means objective found the most efficient way to separate. They are not hidden groups revealed by the data, and there is no further information about age, satisfaction or spending behind them.'),
);

/* segment profile table */
const prow = (label, f, key, fill) => ({ cells: [label, ...SEGS.map(s => f(P[s][key])), f(PO[key])], fill });
const defFill = 'FDF1E6';
add(H2('5.4 The four segments'),
  P_(`${T('profile')} profiles the segments, and ${F('profile_heatmap')} separates what defines a segment (top block) from everything else (bottom block, coloured by percentage difference from the client-base average on a fixed ±10% scale).`, { keepNext: true }),
  tcap('profile', 'Segment profiles. Shaded rows are the attributes that define the segments; every other row is an outcome that the clustering does not separate.'),
  table(['Measure', { t: 'S1', sub: 'Domestic Cash' }, { t: 'S2', sub: 'Domestic Investors' }, { t: 'S3', sub: 'International' }, { t: 'S4', sub: 'Domestic Financed' }, 'All clients'].map(h => (typeof h === 'string' ? h : [h.t, { sub: h.sub }])), [
    { cells: ['Clients', ...SEGS.map(s => n0(P[s].clients)), n0(PO.clients)] },
    prow('Share of clients', p1, 'client_share'),
    prow('Share of revenue', p1, 'revenue_share'),
    prow('US resident', p0, 'pct_usa', defFill), prow('Investment purpose', p0, 'pct_investment', defFill), prow('Loan applied', p0, 'pct_loan', defFill),
    prow('Corporate clients', p1, 'pct_corporate'), prow('Mean age (years)', d1, 'age_mean'), prow('Satisfaction (1–5)', d2, 'satisfaction'), prow('Units per client', d2, 'units_mean'),
    prow('Average unit price', usd, 'avg_price'), prow('Total spend per client', usd, 'spend_mean'), prow('Office share of units', p1, 'office_share'), prow('Buys 5+ units', p1, 'pct_portfolio'),
    prow('Referral: website', p0, 'pct_ref_website'), prow('Referral: agency', p0, 'pct_ref_agency'), prow('Referral: another client', p0, 'pct_ref_client'),
    { cells: ['Cohesion (mean silhouette)', ...SEGS.map(s => d2(D.silhouette_by_segment[s])), d2(PO.mean_silhouette)] },
    { cells: ['Agreement with Ward', ...SEGS.map(s => p0(P[s].ward_agreement)), p0(PO.ward_agreement)] },
  ], [2626, 1280, 1280, 1280, 1280, 1280], { size: 16, align: ['left', 'right', 'right', 'right', 'right', 'right'], boldFirst: false, keep: false }),
  tnote('Percentages may not sum to 100% because of rounding.'),
  ...fig('profile_heatmap', 6.27, 'Segment profiles. Top: the attributes that define the segments, as a share of each segment. Bottom: demographics and behaviour, coloured by the percentage difference from the all-client average and capped at ±10%; apart from the last row the colours are pale. That row compares rates of 2% and 6–8%, which look large in relative terms but rest on small counts (Section 5.5).', 'Two-block heatmap contrasting defining attributes with demographics and behaviour across four segments'),
  P_(`**${NAME.S1}** (${n0(P.S1.clients)} clients, ${p0(P.S1.client_share)}) are US-resident, buy for personal use and applied for no loan. They are the largest segment and the most cohesive (mean silhouette ${d2(D.silhouette_by_segment.S1)}). **${NAME.S2}** (${n0(P.S2.clients)}, ${p0(P.S2.client_share)}) are US-resident investors. ${p0(P.S2.pct_loan)} of them applied for a loan, so the segment mixes leveraged and cash investors, and its cohesion is low (${d2(D.silhouette_by_segment.S2)}). **${NAME.S3}** (${n0(P.S3.clients)}, ${p0(P.S3.client_share)}) live outside the United States. Their purpose and financing mirror the overall client base (${p0(P.S3.pct_investment)} invest, ${p0(P.S3.pct_loan)} borrow), and it is the loosest segment (silhouette ${d2(D.silhouette_by_segment.S3)}) because it groups ten countries that share nothing but residence. **${NAME.S4}** (${n0(P.S4.clients)}, ${p0(P.S4.client_share)}) are US-resident personal-use buyers who applied for a loan.`),
  P_(`Apart from those defining attributes the segments are almost identical. Mean age is ${rngOf('age_mean', d1)} years, satisfaction ${rngOf('satisfaction', d2)}, units per client ${rngOf('units_mean', d2)}, average unit price ${rngOf('avg_price', usdK)} and total spend per client ${rngOf('spend_mean', usdM)}. Each segment's share of revenue is within a point of its share of clients. The one behavioural difference is that ${p1(P.S3.pct_portfolio)} of international clients buy five or more units, against ${rngOf('pct_portfolio', p1, ['S1', 'S2', 'S4'])} in the domestic segments (Section 5.5). Kruskal–Wallis tests find effect sizes (ε²) of ${d3(Math.max(...D.seg_diff.filter(r => r.kind === 'numeric').map(r => r.effect_size)))} or less for every numeric variable (Appendix B). The spend difference reaches p = ${d3(D.seg_diff.find(r => r.variable === 'total_spend').p_value)}, but the gap between the highest and lowest segment is only ${p0((Math.max(...SEGS.map(s => P[s].spend_mean)) - Math.min(...SEGS.map(s => P[s].spend_mean))) / PO.spend_mean)} of average spend, and it would not survive an adjustment for the number of tests.`),
  ...fig('composition', 6.27, 'Composition of each segment by (a) acquisition purpose, (b) loan application, (c) residence and (d) referral channel. The referral mix is the same in every segment.', 'Stacked bars showing purpose, loan, residence and referral channel by segment'),
);

add(H2('5.5 Geography'),
  P_(`The segment mix does not vary across US states (chi-square p = ${d2(G.us_region_seg_p)} over ${G.us_states} states): a client in California is as likely to be a cash home buyer as one in Nevada. Differences between countries in investment share (p = ${d2(G.chi_inv_country_p)}) and loan share (p = ${d2(G.chi_loan_country_p)}) are not statistically significant either, and the confidence intervals in ${F('geography')}b are wide because most country samples are small. The one geographic difference in behaviour is that international clients rarely buy in bulk: ${G.intl_5plus} of ${G.intl_n} (${p1(G.intl_5plus / G.intl_n)}) bought five or more units, against ${G.usa_5plus} of ${n0(G.usa_n)} (${p1(G.usa_5plus / G.usa_n)}) of US clients (Fisher exact p ${pv(G.fisher_p)}). US clients also buy slightly more units (${d2(G.units_usa)} against ${d2(G.units_intl)}) and spend slightly more (${usdM(G.spend_usa)} against ${usdM(G.spend_intl)}), gaps of about 3% that we would not act on.`),
  ...fig('geography', 6.27, 'Geography. (a) The 14 largest regions by client count, split by segment. (b) Investment and loan share by country with 95% Wilson intervals (Wilson, 1927); dotted lines are the overall shares.', 'Stacked bar chart of regions by segment and country bar chart of investment and loan shares with confidence intervals'),
);

add(H2('5.6 Investment profile'),
  P_(`Investors are not a distinct kind of buyer in this data. ${T('invper')} compares the ${n0(INV.investment.n)} clients who bought as an investment with the ${n0(INV.personal.n)} who bought for personal use. They buy the same number of units (${d2(INV.investment.units)} against ${d2(INV.personal.units)}), pay the same price per unit (${usdK(INV.investment.avg_price)} against ${usdK(INV.personal.avg_price)}), spend the same (${usdM(INV.investment.spend)} against ${usdM(INV.personal.spend)}), choose offices at the same rate and use loans at the same rate. None of the differences is statistically significant (all p above ${d2(Math.min(INV.p_spend, INV.p_units, INV.p_price, INV.p_office, INV.p_loan))}).`),
  tcap('invper', 'Investors compared with personal-use buyers.'),
  table(['Measure', 'Investment', 'Personal use', 'p-value'], [
    ['Clients', n0(INV.investment.n), n0(INV.personal.n), '–'],
    ['Units per client', d2(INV.investment.units), d2(INV.personal.units), d2(INV.p_units)],
    ['Average unit price', usd(INV.investment.avg_price), usd(INV.personal.avg_price), d2(INV.p_price)],
    ['Total spend per client', usd(INV.investment.spend), usd(INV.personal.spend), d2(INV.p_spend)],
    ['Office share of units', p1(INV.investment.office), p1(INV.personal.office), d2(INV.p_office)],
    ['Loan applied', p1(INV.investment.loan), p1(INV.personal.loan), d2(INV.p_loan)],
    ['Corporate clients', p1(INV.investment.corp), p1(INV.personal.corp), '–'],
    ['Mean age (years)', d1(INV.investment.age), d1(INV.personal.age), '–'],
    ['Satisfaction (1–5)', d2(INV.investment.sat), d2(INV.personal.sat), '–'],
  ], [3226, 1900, 1900, 2000], { size: 17, align: ['left', 'right', 'right', 'right'] }),
  tnote('p-values are Mann–Whitney tests for numeric measures and a chi-square test for loan share.'),
);

add(H2('5.7 Stability through time'),
  P_(`Each segment\'s share of monthly unit sales is stable across the 24 months (${F('time_and_k5')}a). The step down in 2025 volume therefore hit every segment at about the same rate: units sold fell by ${p0(-(D.desc.sold_drop))} overall, and by between ${p0(-Math.max(...Object.values(D.seg_change)))} and ${p0(-Math.min(...Object.values(D.seg_change)))} in individual segments.`),
  ...fig('time_and_k5', 6.27, '(a) Each segment\'s share of monthly unit sales. (b) The four-segment solution cross-tabulated against a five-cluster solution: the fifth cluster only splits S1 by referral channel.', 'Line chart of segment share of monthly sales and a heatmap comparing the four- and five-cluster solutions'),
);

/* ---------- 6. Personas ---------- */
add(H1('6. Testing the personas in the brief'),
  P_(`The brief suggests four personas. ${T('personas')} states the claim each one implies, the test we used, and the result. ${F('hypotheses')} shows each effect size with a bootstrap 95% interval against the band we treat as negligible.`, { keepNext: true }),
  tcap('personas', 'The four hypothesised personas tested against the data.'),
  table(['Persona', 'Claim tested', 'What the data show', 'Test', 'p', 'Effect', 'Verdict'],
    HY.map(h => [h.persona, h.claim, h.observed, h.test, pv(h.p_value), d3(h.effect) + ' (' + h.effect_name + ')', (h.p_value < 0.05 && Math.abs(h.effect) >= 0.10) ? 'Supported' : 'Not supported']),
    [1250, 1650, 2350, 1350, 600, 900, 926], { size: 15, top: true, boldFirst: true }),
  ...fig('hypotheses', 6.0, 'Effect size for each persona claim with a 95% bootstrap interval (1,000 resamples). The shaded band marks effects smaller than 0.10 in magnitude, which we treat as negligible. Cramér\'s V cannot be negative, so its interval is one-sided.', 'Forest plot of four effect sizes, all inside the negligible band'),
  P_(`**C1, Global Investors.** International clients are no more likely to invest than domestic ones (${HY[0].observed.replace('Investment share: ', '')}; p = ${d2(HY[0].p_value)}). **C2, First-Time Buyers.** Younger clients are not more loan-dependent: clients who applied for a loan are, if anything, a year younger on average, but the difference is not significant (p = ${d2(HY[1].p_value)}), and clients under 35 apply for loans at about the overall rate. First-time status itself is not recorded, and every client bought at least three units in the window, so this test covers only the age and loan part of the persona. **C3, Corporate Buyers.** Corporate clients buy no more units than individuals (${HY[2].observed.split(';')[0].replace('Mean units: ', '')}; p = ${d2(HY[2].p_value)}). **C4, Luxury Investors.** Satisfaction is unrelated to spend (Spearman rho = ${d3(HY[3].effect)}, p = ${d2(HY[3].p_value)}) and to unit price. Income is not recorded, so spend is our only proxy for "luxury".`),
  P_(`Two points make this result informative rather than just disappointing. First, every effect is below 0.05 in magnitude, far under the 0.10 threshold. Second, the tests have adequate power: with ${n0(R.n_clients)} clients a correlation of about 0.06 would be detected 80% of the time, and with ${De.n_corp} corporate clients a difference of about 0.3 standard deviations in units bought would be. If the personas existed at the strength the brief implies, we would have seen them. That does not prove they are absent from Parcl\'s real client base. It shows that the variables that would define them (income, first-time status, the structure of corporate holdings) are missing from this data, and that the variables that are present do not stand in for them.`),
);

/* ---------- 7. Insights ---------- */
add(H1('7. Insights and recommendations'), H2('7.1 Main findings'),
  NUM('n_find', `Parcl\'s buyers fall into four stable groups defined by residence, purpose and financing: ${NAME.S1} (${p0(P.S1.client_share)}), ${NAME.S2} (${p0(P.S2.client_share)}), ${NAME.S3} (${p0(P.S3.client_share)}) and ${NAME.S4} (${p0(P.S4.client_share)}).`),
  NUM('n_find', 'The groups are worth the same. Each segment\'s share of revenue matches its share of clients to within a point, and age, satisfaction, units, price and spend are almost identical across them. The segments say how clients buy, not how much.'),
  NUM('n_find', 'None of the four personas in the brief is visible in the data.'),
  NUM('n_find', `There are no whale accounts. The top 10% of clients hold ${p0(R.top10_share)} of revenue, and the ${R.portfolio_clients} portfolio buyers hold ${p0(R.portfolio_rev_share)}.`),
  NUM('n_find', `International clients almost never buy in bulk (${p1(G.intl_5plus / G.intl_n)} buy five or more units against ${p1(G.usa_5plus / G.usa_n)} of US clients).`),
  NUM('n_find', `Satisfaction is mediocre everywhere. The mean is ${d2(De.sat_mean)} out of 5, ${p0(De.sat_low_share)} of clients scored 1 or 2, and no segment differs from another (Kruskal–Wallis p = ${d2(D.seg_diff.find(r => r.variable === 'satisfaction_score').p_value)}). This looks like a service-level issue, not a segment issue.`),
  NUM('n_find', `Only ${p1(De.ref.Client)} of clients arrived through another client, so referrals are an under-used channel.`),
  NUM('n_find', `Sales volume fell by about a third at the start of 2025 in every segment, while the share of listings that sold and the price per square foot held steady.`),
);

const pb = (id) => P[id];
add(H2('7.2 Segment playbook'),
  P_('The segments differ in intent, financing and geography, so the sensible way to use them is to tailor message, product and channel to those three things. The actions below follow from the profiles. They are hypotheses to test, for example with split campaigns, and not conclusions from the data: the data describe who buys, not what persuades them.', { keepNext: true }),
  tcap('playbook', 'Recommended actions by segment.'),
  table(['Segment', 'Who they are', 'Recommended actions'], [
    [[{ t: 'S1' }, NAME.S1].map(x => (typeof x === 'string' ? x : x.t)), 'US-resident, personal use, no loan. Largest segment.', [
      { bullet: 'Lead with speed and certainty: short closings, transparent pricing, move-in-ready inventory.' },
      { bullet: `Offer multi-unit and office bundles across towers. Every client in the register already bought at least three units, and offices are ${p0(P.S1.office_share)} of this segment\'s purchases.` },
      { bullet: `Seed a referral programme here, but fix service first: satisfaction is ${d2(P.S1.satisfaction)} out of 5.` }]],
    [['S2', NAME.S2], 'US-resident investors. About ' + p0(P.S2.pct_loan) + ' use a loan.', [
      { bullet: 'Lead with investment content: yield, portfolio bundles across towers, alerts for multi-unit blocks.' },
      { bullet: 'Run two message tracks: lender co-marketing for leveraged investors and bulk or portfolio pricing for cash investors.' },
      { bullet: `Keep a named account list of portfolio buyers (5+ units). They are ${p0(R.portfolio_clients / R.n_clients)} of clients but hold ${p0(R.portfolio_rev_share)} of revenue, about one and a half times an average client.` }]],
    [['S3', NAME.S3], 'Resident outside the US, across ten countries. Purpose and financing as in the overall base.', [
      { bullet: 'Build a cross-border service kit: multi-currency pricing, remote closing, local-language material.' },
      { bullet: `Line up cross-border financing partners; ${p0(P.S3.pct_loan)} of these clients apply for a loan.` },
      { bullet: `Do not lead with portfolio offers (${p1(P.S3.pct_portfolio)} buy 5+ units), and message by country, since this is the loosest segment (silhouette ${d2(D.silhouette_by_segment.S3)}).` }]],
    [['S4', NAME.S4], 'US-resident, personal use, loan applied.', [
      { bullet: 'Put mortgage support at first contact: lender partnerships, pre-approval, payment-based price calculators.' },
      { bullet: 'Track time to close and loan-approval turnaround as the operating measures for this group.' },
      { bullet: `Watch satisfaction here. There is no evidence of a gap today (${d2(P.S4.satisfaction)} against ${d2(PO.satisfaction)} overall, p = ${d2(D.seg_diff.find(r => r.variable === 'satisfaction_score').p_value)}), but financing is the likeliest source of friction.` }]],
  ].map(r => [Array.isArray(r[0]) ? [r[0][0] + ' ' + r[0][1]] : r[0], r[1], r[2]]), [1800, 2300, 4926], { size: 16, top: true, boldFirst: true }),
);

add(H2('7.3 Cross-cutting recommendations'),
  BUL(`**Treat satisfaction as a company-wide priority.** With ${p0(De.sat_low_share)} of clients scoring 1 or 2 and no segment standing out, the cause is unlikely to be segment-specific. A short driver survey (price, process, communication, after-sales) would show where to act.`),
  BUL('**Do not budget by segment value.** Segments carry equal revenue per client, so the case for differential spend rests on cost to acquire and serve, which this data does not contain.'),
  BUL(`**Look into the 2025 listing drop.** Sales fell in proportion to listings, and the sell-through rate held at ${p0(R.sold_2025_per_month / R.listings_2025_per_month)}. The question for management is why fewer units were listed.`),
  BUL(`**Build the referral channel.** Referrals bring in ${p1(De.ref.Client)} of clients today. A programme is only likely to work once satisfaction improves.`, { after: 120 }),
);

add(H2('7.4 Data that would make a richer segmentation possible'),
  P_('The personas in the brief are sensible ideas that this dataset cannot test. Collecting the following would allow it:', { keepNext: true }),
  BUL('**Income or net-worth band**, or a proxy such as employer or postcode, for the Global and Luxury Investor personas.'),
  BUL('**Prior property ownership**, so that first-time buyers can be identified. Today every client has three or more purchases, so no first-time buyer can appear.'),
  BUL('**Nationality as well as residence**, to separate expatriates from overseas investors.'),
  BUL('**Financing detail**: loan amount, loan-to-value and approval time, rather than a yes/no flag.'),
  BUL('**Corporate detail**: the number of buying entities, whether units are held or resold, and who the contact person is.'),
  BUL('**Engagement history**: enquiries, viewings and time from first contact to purchase, which would let behavioural segments emerge.', { after: 120 }),
);

/* ---------- 8. Limitations ---------- */
add(H1('8. Limitations'),
  BUL(`**Weak cluster structure.** The silhouette is ${d2(R.silhouette)}, below the 0.26 that marks even weak structure. The segments are stable and useful as a grouping, but they are not natural clusters.`),
  BUL(`**Near-independence of attributes.** A median pairwise association of ${d2(R.assoc_median)} is unusual for client data, where age, geography and purpose normally interact. We recommend confirming with the data owner how the register was compiled, because that determines how far the conclusions carry over to Parcl\'s wider client base.`),
  BUL('**Missing variables.** Income, first-time status and corporate holding structure are not observed, so the corresponding personas could not be tested in full.'),
  BUL('**No occasional buyers.** Every client bought at least three units, so the register contains no single-unit buyers and our behaviour features have limited spread.'),
  BUL('**Corporate ages.** For companies, age describes a contact person and should be read with care.'),
  BUL('**Short window.** Twenty-four months contain one step change in volume and cannot show seasonality or longer cycles.'),
  BUL('**Sold units only.** Behaviour features come from sold units; available listings carry no client.'),
  BUL('**Small samples and multiple tests.** Several countries have fewer than 50 clients, and we report unadjusted p-values. Single-country differences should be read as noise unless they are large.'),
  BUL('**Date-of-birth convention.** The hyphen format was inferred from a sibling file; the effect on age is immaterial (Section 2.3).'),
  BUL('**Map positions.** The dashboard places regions at approximate centroids.'),
  BUL('**Names are labels.** The segment names describe the three defining attributes; they are not behavioural personas.', { after: 120 }),
);

/* ---------- 9. Conclusion ---------- */
add(H1('9. Conclusion'),
  P_(`We combined a client register and a sales ledger into one client-level table and segmented ${n0(R.n_clients)} clients with K-Means, validated against Ward clustering. The data support four stable segments: ${NAME.S1}, ${NAME.S2}, ${NAME.S3} and ${NAME.S4}. They are separated by three categorical attributes (residence, purpose, financing) and are otherwise alike in age, satisfaction and value, and the personas suggested in the brief do not appear. We think that is the honest reading of this dataset, and it is useful in its own right: it tells Parcl that tailoring messaging, financing offers and cross-border services to how clients buy is supported, while allocating effort by expected client value is not. The dashboard that accompanies this paper lets the team explore the segments by country, region, purpose and client type, and the recommendations in Section 7.4 describe the data that would let a future study go further.`),
);

/* ---------- References ---------- */
const REFS = [
  'Caliński, T. and Harabasz, J. (1974). A dendrite method for cluster analysis. //Communications in Statistics//, 3(1), 1–27.',
  'Cohen, J. (1988). //Statistical Power Analysis for the Behavioral Sciences// (2nd ed.). Lawrence Erlbaum Associates.',
  'Cramér, H. (1946). //Mathematical Methods of Statistics//. Princeton University Press.',
  'Davies, D. L. and Bouldin, D. W. (1979). A cluster separation measure. //IEEE Transactions on Pattern Analysis and Machine Intelligence//, 1(2), 224–227.',
  'Hubert, L. and Arabie, P. (1985). Comparing partitions. //Journal of Classification//, 2(1), 193–218.',
  'Kaufman, L. and Rousseeuw, P. J. (1990). //Finding Groups in Data: An Introduction to Cluster Analysis//. Wiley.',
  'Lloyd, S. P. (1982). Least squares quantization in PCM. //IEEE Transactions on Information Theory//, 28(2), 129–137.',
  'MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. //Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability//, 1, 281–297.',
  'Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. //Journal of Machine Learning Research//, 12, 2825–2830.',
  'Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. //Journal of Computational and Applied Mathematics//, 20, 53–65.',
  'Satopää, V., Albrecht, J., Irwin, D. and Raghavan, B. (2011). Finding a "kneedle" in a haystack: detecting knee points in system behavior. //31st International Conference on Distributed Computing Systems Workshops//, 166–171.',
  'Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. //Journal of the American Statistical Association//, 58(301), 236–244.',
  'Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. //Journal of the American Statistical Association//, 22(158), 209–212.',
];
add(H1('References'), ...REFS.map(r => new Paragraph({ children: runs(r, { size: 19 }), spacing: { after: 70 }, indent: { left: 360, hanging: 360 }, alignment: AlignmentType.LEFT })));

/* ---------- Appendices ---------- */
add(new Paragraph({ children: [new PageBreak()] }), H1('Appendix A. Client-level table'),
  P_('The processed table //clients_segmented.csv// has one row per client. Its main columns are:', { keepNext: true }),
  tcap('A1', 'Main columns of the client-level table.'),
  table(['Column', 'Meaning', 'Source'], [
    ['client_id, client_type, gender, country, region', 'Client identifier and attributes as in the register (labels normalised).', 'clients.csv'],
    ['age, age_band', 'Age in years at 31 December 2025, and a five-band grouping.', 'Derived from date of birth'],
    ['acquisition_purpose, loan_applied, referral_channel, satisfaction_score', 'Purpose, loan flag, referral channel and satisfaction as in the register.', 'clients.csv'],
    ['region_group', 'Thirteen-level grouping of region used for clustering.', 'Derived'],
    ['n_units, total_spend, avg_price, avg_ppsf, avg_area', 'Units bought, total and average price, price per sq ft and floor area of sold units.', 'Aggregated from properties.csv'],
    ['share_office, n_towers, portfolio_buyer', 'Share of purchases that are offices, towers bought in, and the 5+ units flag.', 'Aggregated from properties.csv'],
    ['first_purchase, last_purchase, active_months', 'Months of first and last purchase and the span between them.', 'Aggregated from properties.csv'],
    ['segment_id, segment_name, segment_label', 'K-Means segment (S1 is the largest).', 'Model output'],
    ['ward_segment_id, ward_agrees', 'Ward segment mapped to the K-Means segment it overlaps most, and an agreement flag.', 'Model output'],
    ['silhouette, pc1, pc2', 'Client-level silhouette and the first two principal components.', 'Model output'],
    ['*_code', 'Label-encoded versions of the categorical fields.', 'Derived'],
    ['lat, lon, iso3', 'Approximate region centroid and country code for mapping.', 'Derived'],
  ], [3000, 4226, 1800], { size: 16, top: true, boldFirst: true }),
);

const NICE = { age: 'Age', satisfaction_score: 'Satisfaction', n_units: 'Units bought', total_spend: 'Total spend', avg_price: 'Average unit price', share_office: 'Office share', client_type: 'Client type', acquisition_purpose: 'Acquisition purpose', loan_applied: 'Loan applied', referral_channel: 'Referral channel', country: 'Country', gender: 'Gender' };
const DEFINING = ['acquisition_purpose', 'loan_applied', 'country'];
add(H1('Appendix B. Do the segments differ on each variable?'),
  P_('Tests of the variables that define the segments (purpose, loan, country) are circular, since the clustering used them, so they are shown only to quantify how much each variable contributes. The other rows are genuine tests.', { keepNext: true }),
  tcap('B1', 'Differences between the four segments, variable by variable.'),
  table(['Variable', 'Test', 'p-value', 'Effect size', 'Reading'], D.seg_diff.map(r => [NICE[r.variable], r.test, pv(r.p_value), d3(r.effect_size) + ' (' + (r.effect_name === 'epsilon-squared' ? 'ε²' : 'V') + ')',
    DEFINING.includes(r.variable) ? 'Defines the segments (circular)' : ((r.effect_name === 'epsilon-squared' ? r.effect_size < 0.01 : r.effect_size < 0.10) ? 'Negligible' : 'Small or larger')]), [2000, 1800, 1000, 1700, 2526], { size: 17, boldFirst: true }),
  tnote('ε² is the Kruskal–Wallis effect size and V is Cramér\'s V. Conventionally, ε² below 0.01 and V below 0.10 are negligible. The p-values are not adjusted for the twelve tests.'),
);

add(H1('Appendix C. Alternative feature designs'),
  P_('We compared five ways of preparing the features. All use the same 37 columns except design A, which uses only age and satisfaction as numeric features.', { keepNext: true }),
  tcap('C1', 'Clustering quality at k = 4 under five feature designs. Stability uses 30 bootstrap resamples.'),
  table(['Design', 'Silhouette', 'Stability', 'K-Means vs Ward', 'Smallest cluster', 'Best k (silhouette)', 'Comment'], D.design.map((r, i) => [r.design, d3(r.silhouette_k4), d3(r.stability_k4), d3(r.ward_ari_k4), String(r.smallest_k4), `${r.best_k} (${d3(r.silhouette_best)})`,
    ['Similar separation, lower stability, no behaviour features.', 'Highest stability; the design used.', 'Numeric block weighted heavily; algorithms barely agree.', 'Rare categories inflated; tiny cluster.', 'Corporate indicator gets undue leverage.'][i]]),
    [2900, 800, 900, 900, 900, 1000, 1626], { size: 15, top: true, align: ['left', 'right', 'right', 'right', 'right', 'right', 'left'] }),
  tnote('Silhouette, stability, agreement and smallest cluster are all measured at k = 4. The best-silhouette k is the k from 2 to 10 with the highest silhouette for that design; for the two MinMax designs it is k = 10, where clusters approximate exact category combinations.'),
);

add(H1('Appendix D. Reproducing the analysis and running the dashboard'),
  P_('From the project folder:', { keepNext: true }),
  ...['pip install -r requirements.txt', 'python src/pipeline.py             # clean, cluster, validate, export', 'python src/make_figures.py         # figures used in this paper', 'python src/design_comparison.py    # Appendix C', 'python src/export_paper_data.py    # numbers used to build this document', 'node paper/build_paper.js          # rebuild the Word document', 'streamlit run app.py               # launch the dashboard'].map(CODE),
  P_('The dashboard reads the files in data/processed. Its maps load their base layer from the Plotly content-delivery network, so an internet connection is needed for the geography tab. All other views work offline.', { before: 160 }),
);

/* =====================================================================================
   DOCUMENT
   ===================================================================================== */
const hdr = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: LINE, space: 4 } }, children: [new TextRun({ text: 'Buyer Segmentation and Investment Profiling  |  Parcl Co. Limited', size: 16, color: GREYTXT })] })] });
const ftr = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ['Page ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: GREYTXT })] })] });
const emptyH = new Header({ children: [new Paragraph({ children: [] })] }), emptyF = new Footer({ children: [new Paragraph({ children: [] })] });

const doc = new Document({
  creator: 'Author', title: 'Machine Learning based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence',
  description: 'Research paper for Parcl Co. Limited and Unified Mentor',
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 32, bold: true, color: NAVY }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 25, bold: true, color: NAVY }, paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 22, bold: true, color: '333333' }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      ...['n_obj', 'n_find'].map(ref => ({ reference: ref, levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] })),
    ],
  },
  sections: [{
    properties: { titlePage: true, page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } } },
    headers: { default: hdr, first: emptyH }, footers: { default: ftr, first: emptyF },
    children: body,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, 'Buyer_Segmentation_Research_Paper.docx');
  fs.writeFileSync(out, buf); console.log('written', out, (buf.length / 1024).toFixed(0) + ' KB', '| figures placed:', lastFig);
});
