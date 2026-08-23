import os
from datetime import datetime
import pandas as pd
import numpy as np
from django.conf import settings
from django.utils import timezone
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from .models import Transaction, Report

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY        = colors.HexColor("#1c2f4a")
STRIPE      = colors.HexColor("#f2f5f9")
BORDER      = colors.HexColor("#c8d0db")
BLACK       = colors.HexColor("#1a1a1a")
WHITE       = colors.white
MUTED       = colors.HexColor("#555555")

# RFM segment row colours
SEG_BEST    = colors.HexColor("#d4edda")   # Best Customers     → green
SEG_REG     = colors.HexColor("#cce5ff")   # Regular Customers  → blue
SEG_SLIP    = colors.HexColor("#fff3cd")   # Slipping Away      → amber
SEG_LOST    = colors.HexColor("#f8d7da")   # Lost Customers     → red

# Churn risk row colours
RISK_HIGH   = colors.HexColor("#f8d7da")   # Leaving Very Soon  → red
RISK_MED    = colors.HexColor("#fff3cd")   # Needs Attention    → amber
RISK_LOW    = colors.HexColor("#d4edda")   # Still Active       → green

# Cohort cell colours
GREEN_BG    = colors.HexColor("#d4edda")   # >= 50 % returned
AMBER_BG    = colors.HexColor("#fff3cd")   # 25–49 %
RED_BG      = colors.HexColor("#f8d7da")   # < 25 %

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
USABLE = PAGE_W - 2 * MARGIN

RFM_SEG_COLOUR = {
    "Best Customers":    SEG_BEST,
    "Regular Customers": SEG_REG,
    "Slipping Away":     SEG_SLIP,
    "Lost Customers":    SEG_LOST,
}
CHURN_RISK_COLOUR = {
    "Leaving Very Soon":    RISK_HIGH,
    "Needs Your Attention": RISK_MED,
    "Still Active":         RISK_LOW,
}


# ── Shared styles ─────────────────────────────────────────────────────────────
def _st():
    b = getSampleStyleSheet()
    return {
        "title":   ParagraphStyle("t",  fontSize=16, fontName="Helvetica-Bold",
                                  textColor=WHITE, alignment=TA_CENTER, spaceAfter=2),
        "sub":     ParagraphStyle("sb", fontSize=9,  textColor=colors.HexColor("#bbccdd"),
                                  alignment=TA_CENTER, spaceAfter=0),
        "section": ParagraphStyle("sc", fontSize=11, fontName="Helvetica-Bold",
                                  textColor=NAVY, spaceBefore=12, spaceAfter=4),
        "body":    ParagraphStyle("bo", fontSize=9,  textColor=BLACK,
                                  leading=13, spaceAfter=4),
        "th":      ParagraphStyle("th", fontSize=8,  fontName="Helvetica-Bold",
                                  textColor=WHITE, alignment=TA_CENTER),
        "td":      ParagraphStyle("td", fontSize=8,  textColor=BLACK,
                                  alignment=TA_CENTER),
        "td_l":    ParagraphStyle("tl", fontSize=8,  textColor=BLACK,
                                  alignment=TA_LEFT),
        "small":   ParagraphStyle("sm", fontSize=7.5, textColor=MUTED,
                                  alignment=TA_CENTER),
    }


def _header(story, title, lines, st):
    inner_rows = [[Paragraph(title, st["title"])]]
    for line in lines:
        inner_rows.append([Paragraph(line, st["sub"])])
    inner = Table(inner_rows, colWidths=[USABLE])
    inner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(inner)
    story.append(Spacer(1, 8))


def _summary_row(pairs, st):
    n   = len(pairs)
    cw  = USABLE / n
    top = [Paragraph(str(v), ParagraphStyle("sv", fontSize=17,
           fontName="Helvetica-Bold", textColor=NAVY, alignment=TA_CENTER))
           for _, v in pairs]
    bot = [Paragraph(l, st["small"]) for l, _ in pairs]
    tbl = Table([top, bot], colWidths=[cw] * n)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), STRIPE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _section(text, st):
    return [
        Paragraph(text, st["section"]),
        HRFlowable(width=USABLE, thickness=1, color=NAVY, spaceAfter=4),
    ]


def _colour_legend(pairs, st):
    """Small legend row: coloured box + label for each category."""
    cells = []
    for label, bg in pairs:
        box = Table([[""]], colWidths=[8], rowHeights=[8])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX",        (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        cell_content = Table([[box, Paragraph(f"  {label}", st["small"])]],
                             colWidths=[12, 90])
        cell_content.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 2),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
            ("TOPPADDING",    (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        cells.append(cell_content)

    n  = len(cells)
    cw = USABLE / n
    tbl = Table([cells], colWidths=[cw] * n)
    tbl.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _make_table(headers, rows, col_widths, st):
    """Plain striped table (no per-row colour). Used for summary tables."""
    hdr  = [Paragraph(h, st["th"]) for h in headers]
    body = []
    for row in rows:
        body.append([
            Paragraph(str(c), st["td_l"] if j == 0 else st["td"])
            for j, c in enumerate(row)
        ])
    tbl = Table([hdr] + body, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, STRIPE]),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 1), (-1, -1), BLACK),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
    ]))
    return tbl


def _make_coloured_table(headers, rows, col_widths, row_colours, st):
    """
    Table where each data row has an individual background colour.
    row_colours: list of colour objects, one per data row.
    """
    hdr  = [Paragraph(h, st["th"]) for h in headers]
    body = []
    for row in rows:
        body.append([
            Paragraph(str(c), st["td_l"] if j == 0 else st["td"])
            for j, c in enumerate(row)
        ])

    tbl = Table([hdr] + body, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0),  NAVY),
        ("FONTNAME",   (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0),  8),
        ("TEXTCOLOR",  (0, 0), (-1, 0),  WHITE),
        ("ALIGN",      (0, 0), (-1, 0),  "CENTER"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 1), (-1, -1), BLACK),
        ("BOX",        (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",  (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
    ]
    # Apply per-row background colours
    for i, bg in enumerate(row_colours):
        style_cmds.append(("BACKGROUND", (0, i + 1), (-1, i + 1), bg))

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _out_path(label):
    report_dir = os.path.join("media", "reports")
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(report_dir, f"{label}_{ts}.pdf")


def _shop_name(user):
    try:
        return user.shop.shop_name or user.username
    except Exception:
        try:
            return user.shop.owner_name or user.username
        except Exception:
            return user.username


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_dataset(file_path):
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)
    df = pd.read_csv(full_path)
    print("🔥 ORIGINAL COLUMNS:", df.columns.tolist())
    df.columns = df.columns.str.strip()
    column_map = {
        "InvoiceNo":       "InvoiceNo",
        "CustomerID":      "CustomerID",
        "InvoiceDate":     "InvoiceDate",
        "TransactionDate": "InvoiceDate",
        "TotalPrice":      "BillAmount",
        "TotalAmount":     "BillAmount",
        "Amount":          "BillAmount",
        "BillAmount":      "BillAmount",
    }
    df = df.rename(columns=column_map)
    print("🔥 AFTER RENAME:", df.columns.tolist())
    required_cols = ["InvoiceNo", "CustomerID", "InvoiceDate", "BillAmount"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("❌ MISSING COLUMNS:", missing)
        return pd.DataFrame()
    optional_cols = ["CustomerName", "Status"]
    extra_cols    = [c for c in optional_cols if c in df.columns]
    keep_cols     = required_cols + extra_cols
    df = df[keep_cols]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["BillAmount"]  = pd.to_numeric(df["BillAmount"],   errors="coerce")
    df = df.dropna(subset=["InvoiceNo", "CustomerID", "InvoiceDate", "BillAmount"])
    for col in optional_cols:
        if col not in df.columns:
            df[col] = ""
    print("🔥 CLEAN ROWS:", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE RFM
# ─────────────────────────────────────────────────────────────────────────────
def calculate_rfm(user):
    qs = Transaction.objects.filter(user=user)
    if not qs.exists():
        return None

    df    = pd.DataFrame(list(qs.values("customer_id", "transaction_date", "bill_amount")))
    today = timezone.now().date()

    rfm = df.groupby("customer_id").agg(
        Recency  =("transaction_date", lambda x: (today - x.max()).days),
        Frequency=("customer_id",      "count"),
        Monetary =("bill_amount",       "sum"),
    )

    if len(rfm) >= 4:
        features = rfm[["Recency", "Frequency", "Monetary"]].copy()
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        kmeans   = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(X_scaled)
        centroids = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=["Recency", "Frequency", "Monetary"]
        )
        centroids["rank_score"] = (
            centroids["Monetary"].rank()
            + centroids["Frequency"].rank()
            - centroids["Recency"].rank()
        )
        sorted_clusters = centroids["rank_score"].sort_values(ascending=False).index.tolist()
        label_map = {
            sorted_clusters[0]: "Best Customers",
            sorted_clusters[1]: "Regular Customers",
            sorted_clusters[2]: "Slipping Away",
            sorted_clusters[3]: "Lost Customers",
        }
        rfm["Segment"] = rfm["Cluster"].map(label_map)
        rfm["Recency_Score"]  = pd.qcut(rfm["Recency"].rank(method="first"),   q=5, labels=[5,4,3,2,1]).astype(int)
        rfm["Purchase_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5]).astype(int)
        rfm["Spending_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1,2,3,4,5]).astype(int)
        rfm["Overall_Score"]  = rfm["Recency_Score"] + rfm["Purchase_Score"] + rfm["Spending_Score"]
    else:
        rfm["Recency_Score"]  = pd.qcut(rfm["Recency"].rank(method="first"),   q=min(len(rfm),5), labels=False) + 1
        rfm["Purchase_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=min(len(rfm),5), labels=False) + 1
        rfm["Spending_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=min(len(rfm),5), labels=False) + 1
        rfm["Overall_Score"]  = rfm["Recency_Score"].astype(int) + rfm["Purchase_Score"].astype(int) + rfm["Spending_Score"].astype(int)
        def _segment(score):
            if score >= 12: return "Best Customers"
            elif score >= 9: return "Regular Customers"
            elif score >= 6: return "Slipping Away"
            else: return "Lost Customers"
        rfm["Segment"] = rfm["Overall_Score"].apply(_segment)

    rfm = rfm.reset_index().rename(columns={"customer_id": "CustomerID"})
    print(f"✅ RFM calculated for {len(rfm)} customers")
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE CHURN
# ─────────────────────────────────────────────────────────────────────────────
def calculate_churn(user):
    qs = Transaction.objects.filter(user=user)
    if not qs.exists():
        return pd.DataFrame()

    df = pd.DataFrame(list(qs.values("customer_id", "transaction_date", "bill_amount")))
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    ref = df["transaction_date"].max()

    churn = df.groupby("customer_id").agg(
        DaysSinceLastPurchase=("transaction_date", lambda x: (ref - x.max()).days),
        TotalSpend           =("bill_amount",       "sum"),
        Frequency            =("bill_amount",       "count"),
        AvgOrderValue        =("bill_amount",       "mean"),
        FirstPurchase        =("transaction_date",  "min"),
    ).reset_index().rename(columns={"customer_id": "CustomerID"})

    churn["PurchaseSpan"] = (ref - churn["FirstPurchase"]).dt.days

    def _rule_label(d):
        return 2 if d > 120 else (1 if d > 60 else 0)

    churn["_label"] = churn["DaysSinceLastPurchase"].apply(_rule_label)

    feature_cols = ["DaysSinceLastPurchase", "Frequency", "TotalSpend",
                    "AvgOrderValue", "PurchaseSpan"]
    X = churn[feature_cols].fillna(0)
    y = churn["_label"]

    if len(churn) >= 6:
        rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        rf.fit(X, y)
        preds           = rf.predict(X)
        proba           = rf.predict_proba(X)
        high_risk_proba = proba[:, list(rf.classes_).index(2)] \
                          if 2 in rf.classes_ else np.zeros(len(churn))
        label_map = {0: "Still Active", 1: "Needs Your Attention", 2: "Leaving Very Soon"}
        churn["ChurnRisk"]        = [label_map[p] for p in preds]
        churn["ChurnProbability"] = (high_risk_proba * 100).round(1)
    else:
        label_map = {0: "Still Active", 1: "Needs Your Attention", 2: "Leaving Very Soon"}
        churn["ChurnRisk"]        = churn["_label"].map(label_map)
        churn["ChurnProbability"] = churn["DaysSinceLastPurchase"].apply(
            lambda d: round(min(d / 180 * 100, 100), 1)
        )

    churn = churn.drop(columns=["_label", "FirstPurchase"], errors="ignore")
    print(f"✅ Churn calculated for {len(churn)} customers")
    return churn


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATE COHORT
# ─────────────────────────────────────────────────────────────────────────────
def calculate_cohort(user):
    qs = Transaction.objects.filter(user=user)
    if not qs.exists():
        return pd.DataFrame(), {}

    df = pd.DataFrame(list(qs.values("customer_id", "transaction_date")))
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["CohortMonth"] = df.groupby("customer_id")["transaction_date"].transform("min").dt.to_period("M")
    df["OrderMonth"]  = df["transaction_date"].dt.to_period("M")

    cohort_df = (
        df.groupby(["CohortMonth", "OrderMonth"])
          .agg(n_customers=("customer_id", "nunique"))
          .reset_index()
    )

    cohort_df["MonthOffset"] = cohort_df.apply(
        lambda r: (r["OrderMonth"] - r["CohortMonth"]).n, axis=1
    )

    agg = (cohort_df.groupby("MonthOffset")["n_customers"]
                    .sum()
                    .reset_index()
                    .sort_values("MonthOffset"))

    forecast_info = {}
    if len(agg) >= 3:
        X_lr = agg["MonthOffset"].values.reshape(-1, 1)
        y_lr = agg["n_customers"].values.astype(float)
        lr = LinearRegression()
        lr.fit(X_lr, y_lr)
        max_offset    = int(agg["MonthOffset"].max())
        future_months = np.array([max_offset + 1, max_offset + 2]).reshape(-1, 1)
        predictions   = lr.predict(future_months).clip(0)
        base = float(agg.loc[agg["MonthOffset"] == 0, "n_customers"].values[0]) \
               if 0 in agg["MonthOffset"].values else 1
        retention_rates = {
            int(row["MonthOffset"]): round(row["n_customers"] / base * 100, 1)
            for _, row in agg.iterrows()
        }
        forecast_info = {
            "slope":                  round(float(lr.coef_[0]), 2),
            "trend":                  "improving" if lr.coef_[0] > 0 else
                                      ("stable"   if abs(lr.coef_[0]) < 1 else "declining"),
            "forecast_month_offsets": [max_offset + 1, max_offset + 2],
            "forecast_values":        [int(round(p)) for p in predictions],
            "retention_rates":        retention_rates,
        }

    return cohort_df, forecast_info


# ═════════════════════════════════════════════════════════════════════════════
#  RFM REPORT
# ═════════════════════════════════════════════════════════════════════════════
def generate_rfm_report(user, rfm_df):
    file_path = _out_path("RFM_Report")
    doc = SimpleDocTemplate(file_path, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    st    = _st()
    story = []
    shop  = _shop_name(user)
    gen   = datetime.now().strftime("%d %B %Y, %H:%M")

    _header(story, "Who are My Customers", [
        f"Shop: {shop}   |   Owner: {user.username}   |   Generated: {gen}",
    ], st)

    total     = len(rfm_df)
    total_rev = rfm_df["Monetary"].sum()
    avg_ord   = round(rfm_df["Frequency"].mean(), 1)
    avg_rec   = round(rfm_df["Recency"].mean(), 1)

    story.append(_summary_row([
        ("Total Customers",           total),
        ("Total Revenue (Rs.)",       f"{int(total_rev):,}"),
        ("Avg Purchases / Customer",  avg_ord),
        ("Avg Days Since Last Visit", f"{avg_rec}d"),
    ], st))
    story.append(Spacer(1, 10))

    # ── Segment summary ───────────────────────────────────────────────────
    story += _section("Segment Summary", st)
    seg_rows   = []
    seg_colours = []
    for seg in ["Best Customers", "Regular Customers", "Slipping Away", "Lost Customers"]:
        sub = rfm_df[rfm_df["Segment"] == seg]
        if sub.empty:
            continue
        pct = round(len(sub) / total * 100, 1)
        seg_rows.append([
            seg,
            str(len(sub)),
            f"{pct}%",
            f"{round(sub['Recency'].mean(), 1)}d",
            str(round(sub['Frequency'].mean(), 1)),
            f"Rs.{int(sub['Monetary'].mean()):,}",
            f"Rs.{int(sub['Monetary'].sum()):,}",
        ])
        seg_colours.append(RFM_SEG_COLOUR.get(seg, STRIPE))

    story.append(_make_coloured_table(
        ["Segment", "Customers", "Share %", "Avg Days Since Visit",
         "Avg Purchases", "Avg Spend (Rs.)", "Total Revenue (Rs.)"],
        seg_rows,
        [USABLE * w for w in [0.20, 0.09, 0.08, 0.16, 0.12, 0.16, 0.19]],
        seg_colours,
        st
    ))
    story.append(Spacer(1, 6))

    # Legend
    story.append(_colour_legend([
        ("Best Customers",    SEG_BEST),
        ("Regular Customers", SEG_REG),
        ("Slipping Away",     SEG_SLIP),
        ("Lost Customers",    SEG_LOST),
    ], st))
    story.append(Spacer(1, 12))

    # ── Full customer list ────────────────────────────────────────────────
    story += _section(f"Customer List — All {total} Records", st)
    story.append(Spacer(1, 2))

    sorted_df   = rfm_df.sort_values("Monetary", ascending=False)
    cust_rows   = []
    cust_colours = []
    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        seg = str(row["Segment"])
        cust_rows.append([
            str(i),
            str(row["CustomerID"]),
            f"{int(row['Recency'])}d",
            str(int(row["Frequency"])),
            f"Rs.{int(row['Monetary']):,}",
            str(int(row.get("Recency_Score",  0))),
            str(int(row.get("Purchase_Score", 0))),
            str(int(row.get("Spending_Score", 0))),
            str(int(row.get("Overall_Score",  0))),
            seg,
        ])
        cust_colours.append(RFM_SEG_COLOUR.get(seg, STRIPE))

    story.append(_make_coloured_table(
        ["#", "Customer ID", "Days Since Visit", "No. of Purchases",
         "Total Spent (Rs.)", "Visit Score", "Purchase Score", "Spend Score", "Overall Score", "Category"],
        cust_rows,
        [USABLE * w for w in [0.04, 0.10, 0.11, 0.12, 0.13, 0.08, 0.09, 0.08, 0.09, 0.16]],
        cust_colours,
        st
    ))

    doc.build(story)
    Report.objects.create(user=user, report_type="RFM",
                          file=f"reports/{os.path.basename(file_path)}")
    return file_path


# ═════════════════════════════════════════════════════════════════════════════
#  CHURN REPORT
# ═════════════════════════════════════════════════════════════════════════════
def generate_churn_report(user, churn_df):
    file_path = _out_path("Churn_Report")
    doc = SimpleDocTemplate(file_path, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    st    = _st()
    story = []
    shop  = _shop_name(user)
    gen   = datetime.now().strftime("%d %B %Y, %H:%M")

    _header(story, "Who Might Stop Buying", [
        f"Shop: {shop}   |   Owner: {user.username}   |   Generated: {gen}",
    ], st)

    total       = len(churn_df)
    rc          = churn_df["ChurnRisk"].value_counts().to_dict()
    high        = rc.get("Leaving Very Soon",    0)
    medium      = rc.get("Needs Your Attention", 0)
    low         = rc.get("Still Active",         0)
    rev_col     = "TotalSpend" if "TotalSpend" in churn_df.columns else "Monetary"
    rev_at_risk = int(churn_df[churn_df["ChurnRisk"] == "Leaving Very Soon"][rev_col].sum())

    story.append(_summary_row([
        ("Total Customers",       total),
        ("Leaving Very Soon",     high),
        ("Needs Attention",       medium),
        ("Still Active",          low),
        ("Revenue at Risk (Rs.)", f"{rev_at_risk:,}"),
    ], st))
    story.append(Spacer(1, 10))

    # ── Risk breakdown ────────────────────────────────────────────────────
    story += _section("Risk Category Breakdown", st)
    days_col   = "DaysSinceLastPurchase" if "DaysSinceLastPurchase" in churn_df.columns else "Recency"
    risk_rows  = []
    risk_colours = []
    for risk_label in ["Leaving Very Soon", "Needs Your Attention", "Still Active"]:
        sub = churn_df[churn_df["ChurnRisk"] == risk_label]
        if sub.empty:
            continue
        pct = round(len(sub) / total * 100, 1)
        risk_rows.append([
            risk_label,
            str(len(sub)),
            f"{pct}%",
            f"{int(sub[days_col].mean())}d",
            f"Rs.{int(sub[rev_col].sum()):,}",
            f"Rs.{int(sub[rev_col].mean()):,}",
        ])
        risk_colours.append(CHURN_RISK_COLOUR.get(risk_label, STRIPE))

    story.append(_make_coloured_table(
        ["Risk Level", "Customers", "Share %",
         "Avg Days Inactive", "Total Spend (Rs.)", "Avg Spend (Rs.)"],
        risk_rows,
        [USABLE * w for w in [0.24, 0.10, 0.10, 0.16, 0.22, 0.18]],
        risk_colours,
        st
    ))
    story.append(Spacer(1, 6))

    # Legend
    story.append(_colour_legend([
        ("Still Active",         RISK_LOW),
        ("Needs Your Attention", RISK_MED),
        ("Leaving Very Soon",    RISK_HIGH),
    ], st))
    story.append(Spacer(1, 12))

    # ── Full customer list ────────────────────────────────────────────────
    story += _section(f"Customer Return Risk List — All {total} Records", st)
    story.append(Spacer(1, 2))

    prob_col = "ChurnProbability" if "ChurnProbability" in churn_df.columns else None
    sort_col = prob_col if prob_col else days_col

    cust_rows    = []
    cust_colours = []
    for i, (_, row) in enumerate(
            churn_df.sort_values(sort_col, ascending=False).iterrows(), 1):
        risk = str(row["ChurnRisk"])
        r = [
            str(i),
            str(row["CustomerID"]),
            f"{int(row[days_col])}d",
            f"Rs.{int(row[rev_col]):,}",
        ]
        if prob_col:
            r.append(f"{row[prob_col]:.1f}%")
        r.append(risk)
        cust_rows.append(r)
        cust_colours.append(CHURN_RISK_COLOUR.get(risk, STRIPE))

    headers = ["#", "Customer ID", "Days Inactive", "Total Spend (Rs.)"]
    if prob_col:
        headers.append("Leave Risk %")
    headers.append("Risk Level")

    if prob_col:
        cw = [USABLE * w for w in [0.05, 0.15, 0.14, 0.20, 0.14, 0.32]]
    else:
        cw = [USABLE * w for w in [0.05, 0.18, 0.17, 0.24, 0.36]]

    story.append(_make_coloured_table(headers, cust_rows, cw, cust_colours, st))

    doc.build(story)
    Report.objects.create(user=user, report_type="CHURN",
                          file=f"reports/{os.path.basename(file_path)}")
    return file_path


# ═════════════════════════════════════════════════════════════════════════════
#  COHORT REPORT
# ═════════════════════════════════════════════════════════════════════════════
def generate_cohort_report(user, cohort_data):
    if isinstance(cohort_data, tuple):
        cohort_df, forecast_info = cohort_data
    else:
        cohort_df     = cohort_data
        forecast_info = {}

    file_path = _out_path("Cohort_Report")
    doc = SimpleDocTemplate(file_path, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)
    st    = _st()
    story = []
    shop  = _shop_name(user)
    gen   = datetime.now().strftime("%d %B %Y, %H:%M")

    _header(story, "Are Customers Coming Back", [
        f"Shop: {shop}   |   Owner: {user.username}   |   Generated: {gen}",
    ], st)

    if cohort_df is None or cohort_df.empty:
        story.append(Paragraph(
            "No data available. Please upload data covering at least 2 months.",
            st["body"]))
        doc.build(story)
        return file_path

    if "MonthOffset" not in cohort_df.columns:
        cohort_df = cohort_df.copy()
        cohort_df["MonthOffset"] = cohort_df.apply(
            lambda r: (r["OrderMonth"] - r["CohortMonth"]).n, axis=1)

    pivot = cohort_df.pivot_table(
        index="CohortMonth", columns="MonthOffset",
        values="n_customers", aggfunc="sum"
    ).fillna(0).astype(int)

    max_offset   = min(5, int(pivot.columns.max())) if len(pivot.columns) > 0 else 0
    total_visits = int(cohort_df["n_customers"].sum())
    n_cohorts    = pivot.shape[0]
    trend        = forecast_info.get("trend", "N/A").capitalize()
    slope        = forecast_info.get("slope", 0)

    if trend.lower() == "improving":
        trend_label = "Improving ▲"
    elif trend.lower() == "stable":
        trend_label = "Stable ●"
    else:
        trend_label = "Declining ▼"

    story.append(_summary_row([
        ("Months of Data",        n_cohorts),
        ("Total Customer Visits", total_visits),
        ("Return Trend",          trend_label),
        ("Change per Month",      f"{slope:+.0f} customers"),
    ], st))
    story.append(Spacer(1, 10))

    ret_rates  = forecast_info.get("retention_rates", {})
    fc_offsets = forecast_info.get("forecast_month_offsets", [])
    fc_values  = forecast_info.get("forecast_values", [])

    if ret_rates:
        story += _section("Monthly Return Rate", st)
        fc_rows    = []
        fc_colours = []
        for offset in sorted(ret_rates.keys()):
            lbl = "Same Month" if offset == 0 else f"Month +{offset}"
            fc_rows.append([lbl, f"{ret_rates[offset]:.1f}%", "Actual"])
            pct = ret_rates[offset]
            fc_colours.append(GREEN_BG if pct >= 50 else AMBER_BG if pct >= 25 else RED_BG)
        for i, off in enumerate(fc_offsets):
            fc_rows.append([f"Month +{off}", f"{fc_values[i]} customers", "Forecast"])
            fc_colours.append(STRIPE)
        story.append(_make_coloured_table(
            ["Period", "Customers Returned", "Type"],
            fc_rows,
            [USABLE * 0.40, USABLE * 0.35, USABLE * 0.25],
            fc_colours,
            st
        ))
        story.append(Spacer(1, 12))

    story += _section("Monthly Cohort Breakdown", st)
    story.append(_colour_legend([
        ("50%+ returned",          GREEN_BG),
        ("25–49% returned",        AMBER_BG),
        ("Less than 25% returned", RED_BG),
    ], st))
    story.append(Spacer(1, 6))

    hdr_labels = ["First Visit Month", "New Customers"]
    for i in range(1, max_offset + 1):
        hdr_labels.append(f"Month +{i}")

    col_0_w = 32 * mm
    col_1_w = 28 * mm
    rest_w  = (USABLE - col_0_w - col_1_w) / max(max_offset, 1)
    cw      = [col_0_w, col_1_w] + [rest_w] * max_offset

    hdr_row  = [Paragraph(h, st["th"]) for h in hdr_labels]
    tbl_rows = [hdr_row]

    for cohort_month, row in pivot.iterrows():
        base  = int(row.get(0, 0))
        cells = [
            Paragraph(str(cohort_month)[:7], st["td_l"]),
            Paragraph(str(base), st["td"]),
        ]
        for j in range(1, max_offset + 1):
            val = int(row.get(j, 0))
            pct = round(val / base * 100) if base > 0 else 0
            bg  = (GREEN_BG if pct >= 50 else AMBER_BG if pct >= 25 else RED_BG)
            inner = Table([[Paragraph(f"{val} ({pct}%)", st["td"])]],
                          colWidths=[rest_w - 2])
            inner.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, -1), bg),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            cells.append(inner)
        tbl_rows.append(cells)

    cohort_tbl = Table(tbl_rows, colWidths=cw, repeatRows=1)
    cohort_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, STRIPE]),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(cohort_tbl)

    doc.build(story)
    Report.objects.create(user=user, report_type="COHORT",
                          file=f"reports/{os.path.basename(file_path)}")
    return file_path