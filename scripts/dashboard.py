"""
投資部 - スマホで見られるダッシュボードを生成するスクリプト

やること:
1. ウォッチリストの全銘柄を分析する（保有中なら売りシグナル、未保有なら買いシグナルを判定）
2. 結果を1枚の自己完結型HTMLファイル（dashboard.html）として書き出す
3. このHTMLをClaudeのArtifact機能で公開すると、スマホからいつでも見られるURLになる

メールは送らない。発注も自動化しない。あくまで「今どうなっているか」を見るための画面。
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

import config
from signal_check import (
    compute_signals,
    days_since_golden_cross,
    detect_trend_tags,
    fetch_history,
    load_positions,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MARKET_CANDIDATES_FILE = SCRIPT_DIR / "market_candidates.json"

STATUS_LABEL = {
    "buy": "買いシグナル",
    "sell": "売りシグナル",
    "hold": "保有中・様子見",
    "watch": "監視中",
    "error": "取得エラー",
}
STATUS_PRIORITY = {"sell": 0, "buy": 1, "hold": 2, "watch": 3, "error": 4}


def fetch_benchmark_returns():
    """ベータ値計算の基準にする日経平均の日次リターンを取得する"""
    df = fetch_history(config.BENCHMARK_TICKER)
    if df is None:
        return None
    return df["Close"].pct_change().dropna()


def fetch_yf_info(ticker):
    """yfinanceのTicker.infoを1回だけ取得し、ファンダメンタル判定・理論株価・会社概要に使い回す"""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        print(f"[警告] {ticker} のinfo取得に失敗: {e}")
        return {}


def compute_graham_price(yf_info):
    """理論株価（グレアム数）を計算する。sqrt(22.5 × EPS × 1株純資産)という古典的な割安度の目安。
    赤字企業や純資産がマイナスの企業では計算できないため None を返す。"""
    eps = yf_info.get("trailingEps")
    bvps = yf_info.get("bookValue")
    if eps is None or bvps is None or eps <= 0 or bvps <= 0:
        return None
    return (22.5 * eps * bvps) ** 0.5


def fetch_financials(ticker):
    """直近3期分の売上高・営業利益・純利益を取得する（億円換算・小数点第2位まで）。データがなければ空リスト。"""
    try:
        fin = yf.Ticker(ticker).income_stmt
    except Exception as e:
        print(f"[警告] {ticker} の決算データ取得に失敗: {e}")
        return []

    if fin is None or fin.empty:
        return []

    rows = []
    for col in fin.columns:
        revenue = fin.loc["Total Revenue", col] if "Total Revenue" in fin.index else None
        operating = fin.loc["Operating Income", col] if "Operating Income" in fin.index else None
        net = fin.loc["Net Income", col] if "Net Income" in fin.index else None
        if pd.isna(revenue) and pd.isna(operating) and pd.isna(net):
            continue
        rows.append({
            "fiscal_year": col.strftime("%Y年%m月期"),
            "revenue": None if pd.isna(revenue) else round(float(revenue) / 1e8, 2),
            "operating_income": None if pd.isna(operating) else round(float(operating) / 1e8, 2),
            "net_income": None if pd.isna(net) else round(float(net) / 1e8, 2),
        })
        if len(rows) >= 3:
            break
    return rows


def compute_revenue_growth(financials_rows):
    """直近決算（財務諸表の表示と同じデータ）から前期比の売上成長率を計算する。
    yfinanceのrevenueGrowthフィールドは四半期ベースで、表に出す年次決算と数字が食い違うため使わない。"""
    if len(financials_rows) < 2:
        return None
    latest = financials_rows[0].get("revenue")
    previous = financials_rows[1].get("revenue")
    if latest is None or previous is None or previous == 0:
        return None
    return (latest - previous) / previous


def classify_valuation(value, cheap_below, expensive_above):
    if value is None:
        return "算出不可"
    if value <= 0:
        return "算出不可（赤字等）"
    if value < cheap_below:
        return "割安"
    if value <= expensive_above:
        return "標準"
    return "割高"


_JAPANESE_CHAR_RE = re.compile(r"[぀-ヿ一-鿿]")


def fetch_latest_headline(ticker):
    """yfinance経由の直近ニュース見出しを取得する。日本株はほぼ英語しか付かないため、
    日本語の見出しが見つかった場合のみ返す（英語ニュースは表示しない）。"""
    try:
        news = yf.Ticker(ticker).news
    except Exception as e:
        print(f"[警告] {ticker} のニュース取得に失敗: {e}")
        return None
    for item in (news or [])[:10]:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("content", {}).get("title")
        if title and _JAPANESE_CHAR_RE.search(title):
            return title
    return None


YAHOO_FINANCE_BASE = "https://finance.yahoo.co.jp"

# 見出し・本文に含まれるキーワードから、株価への影響を機械的に推測するための辞書
# （AIによる読解ではなく単語一致による判定である点に注意。ただし単純な見出しキーワードだけでなく
# 本文まで見て、かつ数値表現（例:「20%増」）も拾うことで「判断できません」を極力減らしている）
NEWS_NEGATIVE_KEYWORDS = [
    "減益", "減収", "下方修正", "赤字", "損失", "訴訟", "規制強化", "上場廃止",
    "不正", "リコール", "経営破綻", "希薄化", "特別損失", "業績悪化", "自己破産",
    "減配", "無配", "配当見送り", "工場火災", "生産停止", "操業停止", "品質問題",
    "課徴金", "行政処分", "粉飾", "内部統制", "監理銘柄", "特設注意市場",
    "格下げ", "債務超過", "減損", "早期退職", "人員削減", "希望退職",
    "売上未達", "業績下方", "株式売り出し", "公募増資", "第三者割当", "TOB不成立",
    "サイバー攻撃", "情報漏えい", "システム障害", "自主回収",
]
NEWS_POSITIVE_KEYWORDS = [
    "増益", "増収", "上方修正", "最高益", "黒字転換", "提携", "自社株買い",
    "好調", "受注拡大", "増配", "上場来高値", "業務提携", "新製品",
    "特別配当", "自己株式消却", "資本業務提携", "M&A", "買収", "子会社化",
    "新工場", "増産", "設備投資", "大型受注", "特需", "独占", "世界初",
    "格上げ", "黒字幅拡大", "最高値更新", "株式分割", "自己資本比率改善",
    "共同開発", "量産開始", "販売好調", "需要拡大", "認証取得",
]

# 見出し・本文の中の「◯◯%増/減」のような数値表現を検出する正規表現
_PCT_CHANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:程度|ほど|の)?\s*(増|減|高|安)")
# 数値の変化幅がこの値(%)以上なら「大幅」、未満なら「小幅」として強弱を分ける目安
_LARGE_CHANGE_THRESHOLD = 10.0


def analyze_news_impact(title, body=""):
    """見出し＋本文の内容から、株価への影響の考え方をまとめる。
    まずキーワード辞書、次に「◯%増/減」のような数値表現の順にチェックし、
    どちらにも一致しない場合だけ本文の冒頭を要約なしの参考情報として提示する
    （「判断できません」で終わらせず、必ず何らかの手がかりを返す）。"""
    text = f"{title} {body}"

    for kw in NEWS_NEGATIVE_KEYWORDS:
        if kw in text:
            return f"「{kw}」に関する内容。一般的にはマイナス材料になりやすい傾向。"
    for kw in NEWS_POSITIVE_KEYWORDS:
        if kw in text:
            return f"「{kw}」に関する内容。一般的にはプラス材料になりやすい傾向。"

    m = _PCT_CHANGE_RE.search(text)
    if m:
        pct = float(m.group(1))
        direction = m.group(2)
        is_up = direction in ("増", "高")
        scale = "大幅な" if pct >= _LARGE_CHANGE_THRESHOLD else "小幅な"
        if is_up:
            return f"{scale}上昇（{pct:.0f}%{direction}）を示す内容。一般的にはプラス材料になりやすい傾向。"
        return f"{scale}下落（{pct:.0f}%{direction}）を示す内容。一般的にはマイナス材料になりやすい傾向。"

    if body:
        snippet = body[:80].strip()
        if snippet:
            return f"辞書に該当するキーワードはなし。本文冒頭: 「{snippet}…」（内容を読んでご自身で判断してください）"

    return "本文を取得できず、見出しからも株価への影響は判断できませんでした。リンク先で内容を確認してください。"


def fetch_article_body(url, max_chars=600):
    """ニュース記事本文の冒頭部分を取得する（見出しだけでなく実際の内容を分析に使うため）"""
    if not url:
        return ""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[警告] 記事本文の取得に失敗: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.select("article p") or soup.select('div[class*="article"] p') or soup.select("p")
    text = "".join(p.get_text(strip=True) for p in paragraphs)
    return text[:max_chars]


def fetch_yahoo_finance_news(ticker, limit=3):
    """Yahoo!ファイナンスの銘柄別ニュースページから日本語の見出し・リンク・本文・影響考察を取得する
    （yfinanceの英語ニュースだけでは不十分なため、日本語ソースを直接利用する。
    見出しだけでなく実際の記事本文も取得したうえで影響考察を行う）"""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    try:
        resp = requests.get(
            f"{YAHOO_FINANCE_BASE}/quote/{ticker}/news",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[警告] {ticker} のYahoo!ファイナンスニュース取得に失敗: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    headings = soup.select('h3[class*="_NewsItem__heading_"]')

    results = []
    for h in headings[:limit]:
        title = h.get_text(strip=True)
        if not title:
            continue
        link_tag = h.find_parent("a")
        href = link_tag.get("href") if link_tag else None
        url = (YAHOO_FINANCE_BASE + href) if href and href.startswith("/") else href
        body = fetch_article_body(url)
        results.append({"title": title, "url": url, "impact": analyze_news_impact(title, body)})
    return results


RECOMMENDATION_JA = {
    "strong_buy": "強気の買い",
    "buy": "買い",
    "hold": "中立",
    "underperform": "弱気",
    "sell": "売り",
    "none": "評価なし",
}


def build_pro_fundamentals(yf_info, current_price):
    """機関投資家が見るような収益性・財務健全性・アナリスト評価の指標をまとめる（プロ向けファンダメンタル分析）"""

    def pct(key):
        v = yf_info.get(key)
        return None if v is None else v * 100

    roe = pct("returnOnEquity")
    roa = pct("returnOnAssets")
    operating_margin = pct("operatingMargins")
    profit_margin = pct("profitMargins")
    debt_to_equity = yf_info.get("debtToEquity")  # 既に%表記
    dividend_yield = yf_info.get("dividendYield")  # 既に%表記
    peg = yf_info.get("pegRatio")
    current_ratio = yf_info.get("currentRatio")
    target_mean = yf_info.get("targetMeanPrice")
    num_analysts = yf_info.get("numberOfAnalystOpinions")
    recommendation = RECOMMENDATION_JA.get(yf_info.get("recommendationKey"), "不明")
    free_cashflow = yf_info.get("freeCashflow")

    target_gap_pct = None
    if target_mean and current_price:
        target_gap_pct = (target_mean - current_price) / current_price * 100

    return {
        "roe": roe,
        "roa": roa,
        "operating_margin": operating_margin,
        "profit_margin": profit_margin,
        "debt_to_equity": debt_to_equity,
        "dividend_yield": dividend_yield,
        "peg": peg,
        "current_ratio": current_ratio,
        "target_mean": target_mean,
        "target_gap_pct": target_gap_pct,
        "num_analysts": num_analysts,
        "recommendation": recommendation,
        "free_cashflow": free_cashflow,
    }


def build_company_profile(ticker, yf_info, price):
    """会社概要（業種・事業内容・従業員数・時価総額・直近決算・PER/PBR・プロ向け指標・日本語ニュース）をまとめる。
    事業内容は必ず日本語で表示する。"""
    sector_en = yf_info.get("sector")
    sector_ja = config.SECTOR_JA.get(sector_en, sector_en)
    industry_en = yf_info.get("industry")
    industry_ja = config.INDUSTRY_JA.get(industry_en)  # 未登録なら None（英語のまま出さない）

    summary = config.COMPANY_SUMMARIES.get(ticker)

    market_cap = yf_info.get("marketCap")
    employees = yf_info.get("fullTimeEmployees")
    financials = fetch_financials(ticker)
    revenue_growth = compute_revenue_growth(financials)

    per = yf_info.get("trailingPE")
    pbr = yf_info.get("priceToBook")

    news_headlines = fetch_yahoo_finance_news(ticker, limit=3)
    if not news_headlines:
        fallback = fetch_latest_headline(ticker)
        news_headlines = [{"title": fallback, "url": None, "impact": analyze_news_impact(fallback)}] if fallback else []

    return {
        "sector": sector_ja,
        "industry": industry_ja,
        "summary": summary,
        "market_cap": market_cap,
        "employees": employees,
        "revenue_growth": revenue_growth,
        "financials": financials,
        "per": per,
        "pbr": pbr,
        "per_tag": classify_valuation(per, cheap_below=15, expensive_above=25),
        "pbr_tag": classify_valuation(pbr, cheap_below=1, expensive_above=3),
        "news_headlines": news_headlines,
        "pro": build_pro_fundamentals(yf_info, price),
    }


def compute_risk_metrics(df, benchmark_returns):
    """年率リターン・年率ボラティリティ・シャープレシオ・ベータ値を計算する（直近6ヶ月の値動きから）"""
    returns = df["Close"].pct_change().dropna()
    if len(returns) < 20:
        return {"sharpe": None, "beta": None, "volatility": None}

    n = config.TRADING_DAYS_PER_YEAR
    ann_return = float(returns.mean() * n)
    ann_vol = float(returns.std() * (n ** 0.5))
    sharpe = (ann_return - config.RISK_FREE_RATE) / ann_vol if ann_vol > 0 else None

    beta = None
    if benchmark_returns is not None:
        aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner")
        aligned.columns = ["stock", "bench"]
        if len(aligned) > 20 and aligned["bench"].var() > 0:
            beta = float(aligned["stock"].cov(aligned["bench"]) / aligned["bench"].var())

    return {"sharpe": sharpe, "beta": beta, "volatility": ann_vol}


def estimate_days_to_target(df, target_pct, recent_window=10):
    """直近の値動きのペースが続くと仮定した場合、利確ラインに届くまでの目安営業日数を計算する。
    値動きが横ばい・下向きの場合や、目安が長すぎる場合はNoneを返す（見通せないという扱い）。"""
    recent_returns = df["Close"].pct_change().tail(recent_window).dropna()
    if len(recent_returns) < 3:
        return None
    mean_daily = float(recent_returns.mean())
    if mean_daily <= 0:
        return None
    try:
        days = math.log(1 + target_pct) / math.log(1 + mean_daily)
    except (ValueError, ZeroDivisionError):
        return None
    if days <= 0 or days > 250:
        return None
    return days


def analyze_ticker(ticker, positions, benchmark_returns=None):
    name = config.TICKER_NAMES.get(ticker, ticker)
    df = fetch_history(ticker)
    min_len = config.LONG_MA_WINDOW + config.VOLUME_LOOKBACK_WINDOW
    if df is None or len(df) < min_len:
        return {"ticker": ticker, "name": name, "status": "error"}

    df = compute_signals(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(latest["Close"])
    change_pct = (price - float(prev["Close"])) / float(prev["Close"]) * 100

    days_after_cross = days_since_golden_cross(df)
    volume_ratio = None
    if not pd.isna(latest["VOL_AVG"]) and latest["VOL_AVG"] != 0:
        volume_ratio = float(latest["Volume"] / latest["VOL_AVG"])

    # 勢い（トレンドの強さ）: 短期移動平均が長期移動平均よりどれだけ上/下にあるか
    ma_short = latest.get("MA_SHORT")
    ma_long = latest.get("MA_LONG")
    trend_strength = 0.0
    if pd.notna(ma_short) and pd.notna(ma_long) and ma_long != 0:
        trend_strength = float(ma_short / ma_long - 1)
    vr_component = (volume_ratio - 1) * 0.3 if volume_ratio is not None else 0.0
    momentum_score = trend_strength + vr_component

    # 理論的な指標: シャープレシオ・ベータ値・年率ボラティリティ・理論株価（グレアム数）
    risk_metrics = compute_risk_metrics(df, benchmark_returns)
    yf_info = fetch_yf_info(ticker)
    graham_price = compute_graham_price(yf_info)
    graham_gap_pct = None
    if graham_price:
        graham_gap_pct = (price - graham_price) / graham_price * 100
    profile = build_company_profile(ticker, yf_info, price)
    days_to_profit = estimate_days_to_target(df, config.TAKE_PROFIT_PCT)

    info = {
        "ticker": ticker,
        "name": name,
        "price": round(price, 1),
        "change_pct": round(change_pct, 2),
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "days_after_cross": int(days_after_cross) if days_after_cross is not None else None,
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "trend_tags": [],
        "holding": False,
        "trend_strength": trend_strength,
        "momentum_score": momentum_score,
        "sharpe": risk_metrics["sharpe"],
        "beta": risk_metrics["beta"],
        "volatility": risk_metrics["volatility"],
        "graham_price": graham_price,
        "graham_gap_pct": graham_gap_pct,
        "price_history": [round(float(v), 1) for v in df["Close"].tail(60).tolist()],
        "profile": profile,
        "days_to_profit": round(days_to_profit, 1) if days_to_profit is not None else None,
    }

    position = positions.get(ticker)
    if position:
        entry_price = float(position["entry_price"])
        pnl_pct = (price - entry_price) / entry_price * 100

        reasons = []
        lookback = config.GOLDEN_CROSS_RECENT_DAYS + 1
        recent_cross = df["CROSS"].iloc[-lookback:]
        if bool((recent_cross == -1).any()):
            reasons.append("デッドクロス")
        if pnl_pct / 100 <= config.STOP_LOSS_PCT:
            reasons.append("損切りライン")
        if pnl_pct / 100 >= config.TAKE_PROFIT_PCT:
            reasons.append("利確ライン")

        span = config.TAKE_PROFIT_PCT - config.STOP_LOSS_PCT
        gauge_pct = (pnl_pct / 100 - config.STOP_LOSS_PCT) / span
        gauge_pct = max(0.0, min(1.0, gauge_pct))

        info.update({
            "holding": True,
            "entry_price": entry_price,
            "entry_date": position.get("entry_date", ""),
            "pnl_pct": round(pnl_pct, 2),
            "gauge_pct": gauge_pct,
            "status": "sell" if reasons else "hold",
            "reasons": reasons,
            "stop_price": round(entry_price * (1 + config.STOP_LOSS_PCT), 1),
            "target_price": round(entry_price * (1 + config.TAKE_PROFIT_PCT), 1),
        })

        shares = position.get("shares")
        if shares:
            info["shares"] = shares
            info["pnl_yen"] = round((price - entry_price) * shares, 0)

        return info

    is_recent_cross = days_after_cross is not None and days_after_cross <= config.GOLDEN_CROSS_RECENT_DAYS
    is_volume_spike = volume_ratio is not None and volume_ratio >= config.VOLUME_MULTIPLIER
    revenue_growth = profile.get("revenue_growth")
    fundamentals_ok = revenue_growth is None or revenue_growth >= config.FUNDAMENTAL_MIN_REVENUE_GROWTH
    info["fundamentals_ok"] = fundamentals_ok

    if is_recent_cross and is_volume_spike and fundamentals_ok:
        info["status"] = "buy"
        info["trend_tags"] = detect_trend_tags(ticker)
    else:
        info["status"] = "watch"

    # 予算判定: 通常の単元株（100株）を予算2万円以内で買えるか
    # （買えない銘柄はピックアップ対象にしない方針のため、S株での代替案は出さない）
    unit_cost = price * 100
    info["unit_cost"] = round(unit_cost, 0)
    info["fits_unit_budget"] = unit_cost <= config.BUDGET_JPY

    invest_amount = unit_cost if info["fits_unit_budget"] else 0
    info["invest_amount"] = round(invest_amount, 0)
    info["profit_if_target"] = round(invest_amount * config.TAKE_PROFIT_PCT, 0)
    info["loss_if_stop"] = round(invest_amount * config.STOP_LOSS_PCT, 0)

    # 買う場合の目安の損切り価格・利確価格（現在値を基準にした場合の参考値）
    info["stop_price"] = round(price * (1 + config.STOP_LOSS_PCT), 1)
    info["target_price"] = round(price * (1 + config.TAKE_PROFIT_PCT), 1)

    return info


def fmt_price(v):
    return f"{v:,.1f}"


def render_change(change_pct):
    if change_pct is None:
        return ""
    cls = "rise" if change_pct >= 0 else "fall"
    arrow = "▲" if change_pct >= 0 else "▼"
    return f'<span class="change {cls}">{arrow} {abs(change_pct):.2f}%</span>'


def render_tags(tags):
    if not tags:
        return ""
    chips = "".join(f'<span class="tag">{t}</span>' for t in tags)
    return f'<div class="tags">{chips}</div>'


def render_gauge(info):
    pct = info["gauge_pct"] * 100
    pnl = info["pnl_pct"]
    pnl_cls = "rise" if pnl >= 0 else "fall"
    pnl_yen_str = ""
    if "pnl_yen" in info:
        yen = info["pnl_yen"]
        yen_cls = "rise" if yen >= 0 else "fall"
        pnl_yen_str = f'<span class="pnl-yen {yen_cls}">（{yen:+,.0f}円）</span>'
    return f"""
    <div class="gauge">
      <div class="gauge-track">
        <div class="gauge-marker" style="left: {pct:.1f}%"></div>
      </div>
      <div class="gauge-labels">
        <span>ここで売れば損切り（{config.STOP_LOSS_PCT * 100:.0f}%）</span>
        <span class="pnl {pnl_cls}">{pnl:+.2f}%{pnl_yen_str}</span>
        <span>ここで売れば利確（{config.TAKE_PROFIT_PCT * 100:.0f}%）</span>
      </div>
    </div>
    """


def render_budget_box(info):
    """単元株数・予算内かどうか・利益目安・利確までの期間目安をまとめた1ブロック。
    予算入力欄の値が変わるとJSがこの中身を丸ごと再計算して書き換える（data-price/data-days-to-profitを使う）。"""
    price = info["price"]
    unit_cost = info["unit_cost"]
    fits = info["fits_unit_budget"]
    dtp = info.get("days_to_profit")
    period_txt = f"約{dtp:.0f}営業日で" if dtp is not None else "期間は現時点で見通せませんが、"

    lines = f'<div class="budget-line">単元株数: 100株（{unit_cost:,.0f}円）</div>'
    if fits:
        profit = info["profit_if_target"]
        loss = info["loss_if_stop"]
        lines += f'<div class="budget budget-ok">予算{config.BUDGET_JPY:,}円以内で購入可</div>'
        lines += f"""
        <div class="profit-estimate">
          <div class="profit-estimate-title">{unit_cost:,.0f}円分買った場合の目安（{period_txt}利確ラインに届く想定）</div>
          <div class="profit-estimate-row">
            <span class="rise">うまくいけば +{profit:,.0f}円</span>
            <span class="fall">しくじれば {loss:,.0f}円</span>
          </div>
        </div>
        """
    else:
        diff = unit_cost - config.BUDGET_JPY
        lines += f'<div class="budget budget-over">予算オーバー（あと{diff:,.0f}円足りません）</div>'

    dtp_attr = f"{dtp:.1f}" if dtp is not None else ""
    return f'<div class="budget-box" data-price="{price}" data-days-to-profit="{dtp_attr}">{lines}</div>'


def describe_trend_state(info):
    ts = info.get("trend_strength", 0.0)
    vr = info.get("volume_ratio")

    if ts > 0.02:
        trend_txt = f"短期の値動きの平均が長期の平均より{ts * 100:.1f}%上にあり、上昇トレンドの最中"
    elif ts < -0.02:
        trend_txt = f"短期の値動きの平均が長期の平均より{abs(ts) * 100:.1f}%下にあり、下降トレンドの最中"
    else:
        trend_txt = "短期・長期の値動きの平均がほぼ同水準で、方向感に乏しい状態"

    if vr is None:
        vol_txt = "出来高データが不十分"
    elif vr >= 1.3:
        vol_txt = f"出来高も普段の{vr:.2f}倍に増えており、関心が高まっている"
    elif vr <= 0.8:
        vol_txt = f"出来高は普段より少なめ（普段の{vr:.2f}倍）で、あまり注目されていない"
    else:
        vol_txt = f"出来高は普段並み（{vr:.2f}倍）"

    return f"{trend_txt}。{vol_txt}。"


def describe_outlook(info):
    ts = info.get("trend_strength", 0.0)
    vr = info.get("volume_ratio") or 0
    if ts > 0.02 and vr >= 1.3:
        return "上昇が続く可能性がある。ただし出来高を伴った急な動きは反転も早いため、深追いは禁物。"
    if ts > 0.02 and vr < 1.3:
        return "上昇の兆しはあるが出来高が伴っておらず、勢いが続くかどうかは不透明。"
    if ts < -0.02:
        return "下降トレンドが継続する可能性があり、今から買うのはリスクが高い局面。"
    return "方向感が定まっておらず、もう少し様子を見たい局面。"


def describe_fundamentals(info):
    """売上成長率・理論株価との乖離・業種から、ファンダメンタル面の状況を短くまとめる"""
    profile = info.get("profile") or {}
    parts = []

    rg = profile.get("revenue_growth")
    if rg is not None:
        if rg >= 0.05:
            parts.append(f"売上成長率{rg * 100:+.1f}%と伸びている")
        elif rg >= config.FUNDAMENTAL_MIN_REVENUE_GROWTH:
            parts.append(f"売上成長率{rg * 100:+.1f}%でほぼ横ばい")
        else:
            parts.append(f"売上成長率{rg * 100:+.1f}%と縮小しており要注意")
    else:
        parts.append("売上成長率データなし（新興・赤字企業に多い）")

    gap = info.get("graham_gap_pct")
    if gap is not None:
        if gap > 50:
            parts.append(f"理論株価より{gap:+.0f}%割高（将来の成長期待が織り込まれている可能性）")
        elif gap < -20:
            parts.append(f"理論株価より{gap:+.0f}%割安")
        else:
            parts.append(f"理論株価に近い水準（{gap:+.0f}%）")
    else:
        parts.append("理論株価は算出不可")

    per = profile.get("per")
    pbr = profile.get("pbr")
    if per or pbr:
        val_bits = []
        if per:
            val_bits.append(f"PER{per:.1f}倍（{profile.get('per_tag')}）")
        if pbr:
            val_bits.append(f"PBR{pbr:.2f}倍（{profile.get('pbr_tag')}）")
        parts.append("、".join(val_bits))

    pro = profile.get("pro") or {}
    if pro.get("roe") is not None:
        parts.append(f"ROE(自己資本利益率){pro['roe']:.1f}%")
    if pro.get("recommendation") and pro.get("num_analysts"):
        parts.append(f"アナリスト評価は「{pro['recommendation']}」（{pro['num_analysts']}人平均）")
    if pro.get("target_gap_pct") is not None:
        parts.append(f"アナリスト目標株価は現在値より{pro['target_gap_pct']:+.0f}%")

    sector = profile.get("sector")
    if sector:
        parts.append(f"業種は{sector}")

    headlines = profile.get("news_headlines") or []
    if headlines:
        parts.append(f'最近のニュース: 「{headlines[0]["title"]}」（{headlines[0]["impact"]}）')

    return "、".join(parts) + "。"


def render_narrative(info):
    """現状・近々の見通し・根拠をテクニカル指標から機械的に組み立てて表示する（予測を保証するものではない）"""
    current = describe_trend_state(info)

    if info["holding"] and info["status"] == "sell":
        outlook = "売却条件に達しています。あらかじめ決めたルール通りなら、決済を検討する場面です。"
    else:
        outlook = describe_outlook(info)

    basis_parts = [f"勢いスコア {info.get('momentum_score', 0):+.3f}"]
    vr = info.get("volume_ratio")
    basis_parts.append(f"出来高{vr:.2f}倍" if vr is not None else "出来高データなし")
    if info.get("holding"):
        basis_parts.append(f"購入からの含み損益 {info['pnl_pct']:+.2f}%")
    else:
        dac = info.get("days_after_cross")
        if dac is not None:
            basis_parts.append(f"上昇サイン(ゴールデンクロス)から{dac}営業日")
        if info.get("trend_tags"):
            basis_parts.append(f"注目テーマ: {'・'.join(info['trend_tags'])}")
        if info.get("fundamentals_ok") is False:
            basis_parts.append("売上成長率が基準を下回っており要注意")
    basis = "・".join(basis_parts)
    fundamentals_txt = describe_fundamentals(info)

    return f"""
    <div class="narrative">
      <div class="narrative-row"><span class="narrative-label">現状（テクニカル）</span><span>{current}</span></div>
      <div class="narrative-row"><span class="narrative-label">近々の見通し</span><span>{outlook}</span></div>
      <div class="narrative-row"><span class="narrative-label">ファンダメンタル</span><span>{fundamentals_txt}</span></div>
      <div class="narrative-row"><span class="narrative-label">根拠</span><span class="narrative-basis">{basis}</span></div>
    </div>
    """


def render_timing(info):
    """いつ買う/いつ売るべきかの目安を具体的な価格つきで示す"""
    stop_price = info["stop_price"]
    target_price = info["target_price"]

    if info["holding"]:
        sell_txt = (
            f"{stop_price:,.0f}円を下回ったら損切り、{target_price:,.0f}円を上回ったら利確が目安。"
            "下降サイン（デッドクロス）が出た場合もそこが売りどきの合図。"
        )
        return f"""
        <div class="timing">
          <div class="timing-row"><span class="timing-label">売るタイミング</span><span>{sell_txt}</span></div>
        </div>
        """

    price = info["price"]
    if info["status"] == "buy":
        buy_txt = f"買いシグナルが成立済み。狙うなら現在値{price:,.1f}円付近が目安。"
    else:
        buy_txt = (
            "まだ正式な買いシグナルではない（出来高が普段の1.5倍に届いていない、"
            "または上昇サインの発生から日数が経ちすぎている）。狙うなら「出来高が急に増え、"
            f"直近3営業日以内に上昇サインが出た」タイミングが目安。今すぐ試すなら現在値{price:,.1f}円付近。"
        )
    sell_txt = f"買った場合は{stop_price:,.0f}円を下回ったら損切り、{target_price:,.0f}円を上回ったら利確が目安。"

    dtp = info.get("days_to_profit")
    if dtp is not None:
        period_txt = (
            f"直近10営業日のペースが続くと仮定すると、利確ライン(+{config.TAKE_PROFIT_PCT * 100:.0f}%)まで"
            f"およそ{dtp:.0f}営業日の目安（勢いが続く保証はありません）。"
        )
    else:
        period_txt = "直近の値動きが横ばい・下向きのため、利確までの期間は現時点では見通せません。"

    return f"""
    <div class="timing">
      <div class="timing-row"><span class="timing-label">買うタイミング</span><span>{buy_txt}</span></div>
      <div class="timing-row"><span class="timing-label">売るタイミング</span><span>{sell_txt}</span></div>
      <div class="timing-row"><span class="timing-label">期間の目安</span><span>{period_txt}</span></div>
    </div>
    """


def render_pickup(results):
    """予算内（通常100株が2万円以内）で「今、何を買うべきか」の一押し銘柄を表示する。
    100株を2万円以内で買えない銘柄はそもそも候補にしない。"""
    affordable = [r for r in results if r.get("fits_unit_budget")]
    buy_candidates = [r for r in affordable if r["status"] == "buy"]
    watch_candidates = [r for r in affordable if r["status"] == "watch" and "momentum_score" in r]

    if buy_candidates:
        pool = sorted(buy_candidates, key=lambda r: r["momentum_score"], reverse=True)
        heading = "買いシグナル銘柄からの一押し（予算内で買えるものだけ）"
    elif watch_candidates:
        pool = sorted(watch_candidates, key=lambda r: r["momentum_score"], reverse=True)[:3]
        heading = "本日はシグナル未発生。次点候補（勢いスコア順・予算内のみ）"
    else:
        return """
        <section class="pickup-zone">
          <div class="pickup-card">
            <p class="pickup-heading">予算2万円・通常100株で買える候補は今のところありません</p>
          </div>
        </section>
        """

    budget = config.BUDGET_JPY
    rows = ""
    for r in pool:
        leftover = budget - r["unit_cost"]
        rows += f"""
        <div class="pickup-row">
          <span class="pickup-name">{r['name']}<span class="ticker">{r['ticker']}</span></span>
          <span class="pickup-fit">100株={r['unit_cost']:,.0f}円（残り{leftover:,.0f}円）<br>
            <span class="pickup-profit"><span class="rise">+{r['profit_if_target']:,.0f}円</span> / <span class="fall">{r['loss_if_stop']:,.0f}円</span></span>
          </span>
        </div>
        """

    return f"""
    <section class="pickup-zone">
      <h2 class="section-title">予算{budget:,}円で今どうする</h2>
      <div class="pickup-card">
        <p class="pickup-heading">{heading}</p>
        {rows}
      </div>
    </section>
    """


def render_glossary():
    terms = [
        ("ゴールデンクロス", "短期間の値動きの平均線が、長期間の平均線を下から上に追い越すこと。株価が上向きに変わり始めたサインとされる。"),
        ("デッドクロス", "ゴールデンクロスの逆。短期の平均線が長期の平均線を上から下に割り込むこと。株価が下向きに変わり始めたサインとされる。"),
        ("取引の活発さ（出来高）", "その日に売買が成立した株数のこと。「普段の1.5倍」なら、いつもより多くの人がその株を売買していて注目度が上がっている合図。"),
        ("勢いスコア", "上昇トレンドの強さと取引の活発さを組み合わせた、このダッシュボード独自の目安。プラスが大きいほど「今、上昇の勢いが強い」ことを意味する。正式な売買シグナルではなく、あくまで順位付けの参考値。"),
        ("損切り／利確ライン", "損切り＝あらかじめ決めた金額まで下がったら、それ以上損をしないよう売って損失を確定させるルール。利確＝決めた金額まで上がったら、利益を確定させるために売るルール。"),
        ("理論株価（グレアム数）", "企業の利益（EPS）と純資産（1株あたり純資産）から計算する、割安/割高の古典的な目安。"
         "sqrt(22.5 × EPS × 1株純資産)という式で計算する。"
         "半導体・AI関連株のようにPER（株価収益率）が高い成長株は、現在値がこの理論株価を数倍〜数十倍上回るのが普通で、"
         "それ自体が「即・危険」を意味するわけではない（市場が将来の成長を織り込んでいるだけの場合も多い）。"
         "赤字企業や純資産がマイナスの企業では計算自体ができず、その場合は「算出不可」と表示される。"),
        ("シャープレシオ", "「リスクの割に、どれだけ効率よくリターンを得ているか」を示す数値。直近6ヶ月の値動きから計算。"
         "プラスが大きいほど値動きの荒さの割にリターンが良く、マイナスは無リスク資産（国債など）に負けていることを意味する。"
         "一般的に1以上で優秀とされる。無リスク金利は日本の10年国債利回り(年2.7%と仮定)を使用。"
         "カード上の「優秀／普通／劣後」というバッジは、この数値をひと目でわかるように色分けしたもの。"),
        ("ベータ値", "日経平均株価と比べて、株価がどれだけ大きく動くかを示す数値。1より大きいと日経平均より値動きが激しく、"
         "1より小さいと値動きが穏やか。マイナスなら日経平均と逆方向に動く傾向がある。"
         "カード上の「穏やか／平均的／大荒れ」というバッジで色分けして表示している。"),
        ("年率ボラティリティ", "値動きの荒さを1年あたりに換算した数値。大きいほどハイリスク・ハイリターンな値動きをする銘柄。"),
        ("🔍全上場銘柄スキャン", "ウォッチリストに入れていない銘柄も含め、東証プライム・スタンダード・グロース市場の"
         "全内国株式（約3,700銘柄）を対象に、ゴールデンクロス＋出来高急増の条件だけで探した候補。"
         "件数が多いと重いため、ファンダメンタルや理論株価などの詳しい分析は上位の銘柄のみに絞って表示している。"
         "このスキャン自体は毎回のダッシュボード更新では実行されず、市場が閉まった後などに時々（週1回程度）実行する想定。"),
        ("「現状」の上昇/下降トレンドと前日比の違い", "カードの「現状」欄が示すトレンドは、直近5日間と25日間の平均を比べた数週間単位の傾向。"
         "価格の横にある前日比（▲▼のパーセント）は1日だけの値動きなので、この2つが逆方向を示すことがある"
         "（例: 数週間単位では上昇中でも、今日だけたまたま少し下がった、など）。"),
        ("利確までの期間の目安", "直近10営業日の値動きのペースがそのまま続くと仮定した場合、利確ライン"
         f"（+{config.TAKE_PROFIT_PCT * 100:.0f}%）に届くまでの営業日数を機械的に計算したもの。"
         "値動きのペースが今後も続く保証はまったくないため、あくまで参考値。値動きが横ばい・下降中の銘柄は「見通せません」と表示される。"),
        ("ポートフォリオの品質ゲート", f"「予算内で買える」というだけでなく、シャープレシオが{config.MIN_SHARPE_FOR_PORTFOLIO:.1f}以上、"
         f"かつ利確までの目安が{config.MAX_DAYS_TO_PROFIT_FOR_PORTFOLIO}営業日以内の銘柄だけを"
         "おすすめポートフォリオ計算の候補にしている。リスクに見合わない銘柄や、利益化まで時間がかかりすぎる銘柄は"
         "最初から候補から除外される。"),
        ("短期×がっつり／中長期×着実", "各候補を「時間軸（利確までの目安が"
         f"{config.SHORT_TERM_DAYS}営業日以内なら短期、それ以上なら中長期）」と"
         f"「値動きの強さ（年率ボラティリティが{config.AGGRESSIVE_VOL_THRESHOLD * 100:.0f}%以上ならがっつり、"
         "それ未満なら着実）」の2軸で分類したもの。方針として「短期×がっつり」型を最優先で組み入れつつ、"
         "他のタイプも組み合わせて表示している。"),
        ("PER（株価収益率）", "株価が1株あたり利益の何倍まで買われているかを示す数値。低いほど「利益の割に株価が安い＝割安」とされる目安。"
         "一般的な目安として15倍未満を「割安」、15〜25倍を「標準」、25倍超を「割高」タグで表示（業種によって適正水準は大きく異なるため、あくまで大まかな目安）。赤字企業は算出不可。"),
        ("PBR（株価純資産倍率）", "株価が1株あたり純資産（会社を清算した場合の取り分の目安）の何倍まで買われているかを示す数値。"
         "1倍未満なら理論上は純資産より株価が安い状態。一般的な目安として1倍未満を「割安」、1〜3倍を「標準」、3倍超を「割高」タグで表示。"),
    ]
    rows = "".join(
        f'<div class="glossary-row"><span class="glossary-term">{term}</span>'
        f'<span class="glossary-desc">{desc}</span></div>'
        for term, desc in terms
    )
    return f"""
    <section class="glossary">
      <h2 class="section-title">このページの用語</h2>
      <div class="glossary-list">{rows}</div>
    </section>
    """


def render_quant(info):
    """理論株価・シャープレシオ・ベータ値・ボラティリティを、数値+視覚的なバッジ/ゲージで表示する"""
    graham = info.get("graham_price")
    graham_txt = f'{graham:,.0f}円' if graham else "算出不可（赤字、または純資産データなし）"
    graham_gauge = render_graham_gauge(info) if graham else ""

    sharpe = info.get("sharpe")
    if sharpe is None:
        sharpe_badge = '<span class="metric-badge badge-muted">算出不可</span>'
    elif sharpe >= 1:
        sharpe_badge = f'<span class="metric-badge badge-good">優秀 {sharpe:+.2f}</span>'
    elif sharpe >= 0:
        sharpe_badge = f'<span class="metric-badge badge-neutral">普通 {sharpe:+.2f}</span>'
    else:
        sharpe_badge = f'<span class="metric-badge badge-bad">劣後 {sharpe:+.2f}</span>'

    beta = info.get("beta")
    if beta is None:
        beta_badge = '<span class="metric-badge badge-muted">算出不可</span>'
    elif beta >= 1.3:
        beta_badge = f'<span class="metric-badge badge-bad">大荒れ {beta:.2f}</span>'
    elif beta >= 0.7:
        beta_badge = f'<span class="metric-badge badge-neutral">平均的 {beta:.2f}</span>'
    else:
        beta_badge = f'<span class="metric-badge badge-good">穏やか {beta:.2f}</span>'

    vol = info.get("volatility")
    vol_txt = f"{vol * 100:.1f}%" if vol is not None else "算出不可"

    return f"""
    <div class="quant">
      <div class="quant-row"><span class="quant-label">理論株価（グレアム数）</span><span>{graham_txt}</span></div>
      {graham_gauge}
      <div class="quant-grid">
        <div><span class="quant-label">シャープレシオ</span>{sharpe_badge}</div>
        <div><span class="quant-label">ベータ値</span>{beta_badge}</div>
        <div><span class="quant-label">年率ボラティリティ</span><span class="quant-value">{vol_txt}</span></div>
      </div>
    </div>
    """


def to_tradingview_symbol(ticker):
    """yfinance形式のティッカー（例: 6857.T）をTradingViewのシンボル形式（例: TSE:6857）に変換する"""
    code = ticker.replace(".T", "")
    return f"TSE:{code}"


def render_tradingview_widget(ticker):
    """TradingView社が無料公開している埋め込みウィジェットで、本物のリアルタイムチャートを表示する。
    ページを開いている間、TradingView側から直接ライブの値動きが流れてくる（当サイトのデータ取得とは無関係）。
    ダークモード/ライトモードは、このページの配色設定(data-theme)に合わせて自動で切り替える。"""
    symbol = to_tradingview_symbol(ticker)
    widget_id = "tv_" + re.sub(r"[^a-zA-Z0-9]", "_", ticker)
    return f"""
    <div class="tv-widget-container" id="{widget_id}"></div>
    <script>
    (function() {{
      var root = document.documentElement;
      var theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      var forced = root.getAttribute('data-theme');
      if (forced === 'dark' || forced === 'light') theme = forced;
      var el = document.getElementById('{widget_id}');
      var s = document.createElement('script');
      s.src = 'https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js';
      s.async = true;
      s.text = JSON.stringify({{
        symbol: '{symbol}',
        width: '100%',
        height: 160,
        locale: 'ja',
        dateRange: '1D',
        colorTheme: theme,
        isTransparent: true,
        autosize: true
      }});
      el.appendChild(s);
    }})();
    </script>
    """


def render_sparkline(info):
    """直近の値動きを線グラフで表示する。理論株価があれば破線の目安ラインも重ねる。"""
    prices = info.get("price_history")
    if not prices or len(prices) < 5:
        return ""

    width, height = 280, 64
    pad_x, pad_y = 4, 10
    graham = info.get("graham_price")

    lo, hi = min(prices), max(prices)
    if graham:
        lo = min(lo, graham)
        hi = max(hi, graham)
    span = (hi - lo) or 1

    n = len(prices)
    step = (width - 2 * pad_x) / max(n - 1, 1)

    def px(i):
        return pad_x + i * step

    def py(v):
        return height - pad_y - (v - lo) / span * (height - 2 * pad_y)

    points = [(px(i), py(v)) for i, v in enumerate(prices)]
    path_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    area_d = (
        path_d
        + f" L {points[-1][0]:.1f} {height - pad_y:.1f}"
        + f" L {points[0][0]:.1f} {height - pad_y:.1f} Z"
    )

    rising = prices[-1] >= prices[0]
    line_color = "var(--rise)" if rising else "var(--fall)"

    graham_line = ""
    if graham and lo <= graham <= hi:
        gy = py(graham)
        graham_line = (
            f'<line x1="{pad_x}" y1="{gy:.1f}" x2="{width - pad_x}" y2="{gy:.1f}" '
            f'stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="3,3" />'
            f'<text x="{width - pad_x}" y="{max(gy - 3, 8):.1f}" text-anchor="end" '
            f'class="spark-label">理論株価</text>'
        )

    last_x, last_y = points[-1]

    return f"""
    <div class="sparkline">
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none"
        role="img" aria-label="直近{len(prices)}日間の株価推移">
        <path d="{area_d}" fill="{line_color}" opacity="0.12" stroke="none" />
        <path d="{path_d}" fill="none" stroke="{line_color}" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round" />
        {graham_line}
        <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.2" fill="{line_color}"
          stroke="var(--surface)" stroke-width="1.5" />
      </svg>
      <div class="sparkline-caption"><span>{len(prices)}営業日前</span><span>今日</span></div>
    </div>
    """


def render_graham_gauge(info):
    """理論株価に対して現在値がどこにあるかを視覚的なバーで示す"""
    gap = info.get("graham_gap_pct")
    if gap is None:
        return ""
    # -80%(理論株価の1/5程度)〜+300%(理論株価の4倍)を表示範囲とし、その中に収める（範囲外は端で飽和表示）
    clamped = max(-80.0, min(300.0, gap))
    pct = (clamped + 80) / 380 * 100
    cls_style = "color: var(--sell);" if gap > 0 else "color: var(--buy);"
    return f"""
    <div class="gauge graham-gauge">
      <div class="gauge-track graham-track">
        <div class="gauge-marker" style="left: {pct:.1f}%"></div>
      </div>
      <div class="gauge-labels">
        <span>割安</span>
        <span style="font-weight: 700; {cls_style}">理論株価より{gap:+.0f}%</span>
        <span>割高</span>
      </div>
    </div>
    """


def fmt_market_cap(v):
    if not v:
        return "不明"
    oku = v / 1e8
    if oku >= 10000:
        return f"{oku / 10000:.1f}兆円"
    return f"{oku:,.0f}億円"


def render_financials_table(rows):
    if not rows:
        return '<p class="financials-empty">決算データを取得できませんでした。</p>'

    def cell(v):
        return f"{v:,.2f}" if v is not None else "-"

    body = ""
    for r in rows:
        body += (
            f"<tr><td>{r['fiscal_year']}</td>"
            f"<td>{cell(r['revenue'])}</td>"
            f"<td>{cell(r['operating_income'])}</td>"
            f"<td>{cell(r['net_income'])}</td></tr>"
        )

    return f"""
    <div class="financials-table-wrap">
      <table class="financials-table">
        <thead><tr><th>決算期</th><th>売上高</th><th>営業利益</th><th>純利益</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
      <p class="financials-unit">単位: 億円（直近が上）</p>
    </div>
    """


def render_pro_fundamentals(pro):
    """ROE・ROA・利益率・財務健全性・アナリスト評価など、プロが見る指標をまとめて表示する"""
    if not pro:
        return ""

    def pctcell(v):
        return f"{v:.1f}%" if v is not None else "-"

    roe = pctcell(pro.get("roe"))
    roa = pctcell(pro.get("roa"))
    op_margin = pctcell(pro.get("operating_margin"))
    profit_margin = pctcell(pro.get("profit_margin"))
    dte = f"{pro['debt_to_equity']:.0f}%" if pro.get("debt_to_equity") is not None else "-"
    div_yield = f"{pro['dividend_yield']:.2f}%" if pro.get("dividend_yield") is not None else "配当なし/不明"
    peg = f"{pro['peg']:.2f}" if pro.get("peg") is not None else "-"
    current_ratio = f"{pro['current_ratio']:.2f}" if pro.get("current_ratio") is not None else "-"
    fcf = fmt_market_cap(pro.get("free_cashflow")) if pro.get("free_cashflow") is not None else "-"
    if pro.get("free_cashflow") is not None and pro["free_cashflow"] < 0:
        fcf = f"-{fcf}"

    target_html = ""
    if pro.get("target_mean") is not None:
        gap = pro.get("target_gap_pct")
        gap_html = f'<span class="{"rise" if gap and gap >= 0 else "fall"}">（現在値より{gap:+.0f}%）</span>' if gap is not None else ""
        target_html = f"<span>アナリスト目標株価: {pro['target_mean']:,.0f}円{gap_html}</span>"

    rec_html = ""
    if pro.get("recommendation") and pro.get("recommendation") != "不明":
        n = pro.get("num_analysts")
        n_txt = f"（アナリスト{n}人）" if n else ""
        rec_html = f"<span>アナリスト評価: <strong>{pro['recommendation']}</strong>{n_txt}</span>"

    return f"""
    <p class="financials-title">📊 プロ向け指標（収益性・財務健全性・アナリスト評価）</p>
    <div class="pro-fund-grid">
      <div><span class="quant-label">ROE（自己資本利益率）</span><span class="quant-value">{roe}</span></div>
      <div><span class="quant-label">ROA（総資産利益率）</span><span class="quant-value">{roa}</span></div>
      <div><span class="quant-label">営業利益率</span><span class="quant-value">{op_margin}</span></div>
      <div><span class="quant-label">純利益率</span><span class="quant-value">{profit_margin}</span></div>
      <div><span class="quant-label">負債比率（対自己資本）</span><span class="quant-value">{dte}</span></div>
      <div><span class="quant-label">配当利回り</span><span class="quant-value">{div_yield}</span></div>
      <div><span class="quant-label">PEGレシオ</span><span class="quant-value">{peg}</span></div>
      <div><span class="quant-label">流動比率</span><span class="quant-value">{current_ratio}</span></div>
      <div><span class="quant-label">フリーCF</span><span class="quant-value">{fcf}</span></div>
    </div>
    <div class="profile-facts">
      {target_html}
      {rec_html}
    </div>
    """


def render_news_headlines(headlines):
    if not headlines:
        return '<p class="profile-summary-muted">関連ニュースは見つかりませんでした。</p>'

    items = ""
    for h in headlines:
        title = h["title"]
        url = h.get("url")
        impact = h.get("impact", "")
        title_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>' if url else title
        items += f'<li>{title_html}<div class="news-impact">💡 {impact}</div></li>'
    return f'<ul class="news-list">{items}</ul>'


def render_company_profile(info):
    """会社概要（業種・事業内容・時価総額・従業員数・直近決算・プロ向け指標・関連ニュース）をタップで開ける形で表示する。
    事業内容は必ず日本語（未翻訳の銘柄は業種・数値情報のみ）で表示する。"""
    profile = info.get("profile") or {}
    sector = profile.get("sector") or "不明"
    industry = profile.get("industry")
    industry_txt = f"（{industry}）" if industry else ""
    mcap_txt = fmt_market_cap(profile.get("market_cap"))
    employees = profile.get("employees")
    emp_txt = f"{employees:,}人" if employees else "不明"

    per = profile.get("per")
    pbr = profile.get("pbr")
    per_txt = f"{per:.1f}倍" if per else "算出不可"
    pbr_txt = f"{pbr:.2f}倍" if pbr else "算出不可"
    per_tag_cls = {"割安": "tag-cheap", "標準": "tag-normal", "割高": "tag-expensive"}.get(profile.get("per_tag"), "tag-muted")
    pbr_tag_cls = {"割安": "tag-cheap", "標準": "tag-normal", "割高": "tag-expensive"}.get(profile.get("pbr_tag"), "tag-muted")

    summary = profile.get("summary")
    if summary:
        summary_html = f'<p class="profile-summary">{summary}</p>'
    else:
        summary_html = '<p class="profile-summary profile-summary-muted">事業内容の日本語要約は準備中です。上記の業種・決算データを参考にしてください。</p>'

    if not (summary or profile.get("financials") or profile.get("market_cap")):
        return ""

    return f"""
    <details class="more-details company-profile">
      <summary>🏢 この会社について</summary>
      <div class="profile-facts">
        <span>業種: {sector}{industry_txt}</span>
        <span>時価総額: {mcap_txt}</span>
        <span>従業員数: {emp_txt}</span>
      </div>
      <div class="profile-facts">
        <span>PER: {per_txt} <span class="valuation-tag {per_tag_cls}">{profile.get('per_tag', '算出不可')}</span></span>
        <span>PBR: {pbr_txt} <span class="valuation-tag {pbr_tag_cls}">{profile.get('pbr_tag', '算出不可')}</span></span>
      </div>
      {summary_html}
      {render_pro_fundamentals(profile.get("pro"))}
      <p class="financials-title">直近3期の決算</p>
      {render_financials_table(profile.get("financials"))}
      <p class="financials-title">📰 関連ニュース（Yahoo!ファイナンス）</p>
      {render_news_headlines(profile.get("news_headlines"))}
    </details>
    """


def render_card(info):
    status = info["status"]
    label = STATUS_LABEL[status]

    if status == "error":
        return f"""
        <article class="card status-error" data-change="0" data-volume="0" data-momentum="0" data-sharpe="-999"
          data-graham="0" data-status="{STATUS_PRIORITY.get('error', 5)}">
          <div class="card-top">
            <div class="name">{info['name']}<span class="ticker">{info['ticker']}</span></div>
            <span class="chip chip-error">{label}</span>
          </div>
          <p class="error-msg">株価データを取得できませんでした。</p>
        </article>
        """

    change_html = render_change(info.get("change_pct"))
    tags_html = render_tags(info.get("trend_tags"))

    more_html = f"""
    <details class="more-details">
      <summary>詳しく見る（理論指標・現状・タイミング）</summary>
      {render_quant(info)}
      {render_narrative(info)}
      {render_timing(info)}
    </details>
    """

    detail_html = ""
    if info["holding"]:
        reasons = info.get("reasons") or []
        reasons_html = ""
        if reasons:
            reasons_html = f'<div class="reasons">理由: {"、".join(reasons)}</div>'
        detail_html = f"""
        <div class="detail">
          <span>取得価格 {fmt_price(info['entry_price'])}円（{info.get('entry_date', '')}）</span>
        </div>
        {render_gauge(info)}
        {reasons_html}
        {render_company_profile(info)}
        {more_html}
        """
    else:
        vr = info.get("volume_ratio")
        dac = info.get("days_after_cross")
        vr_str = f"普段の{vr:.2f}倍" if vr is not None else "-"
        dac_str = f"{dac}営業日前に上昇サイン" if dac is not None else "上昇サインなし"
        detail_html = f"""
        <div class="detail">
          <span>取引の活発さ: {vr_str}</span>
          <span class="dot-sep">・</span>
          <span>{dac_str}</span>
        </div>
        {tags_html}
        {render_budget_box(info)}
        {render_company_profile(info)}
        {more_html}
        """

    data_change = info.get("change_pct", 0) or 0
    data_volume = info.get("volume_ratio") or 0
    data_momentum = info.get("momentum_score", 0) or 0
    data_sharpe = info.get("sharpe")
    data_sharpe = data_sharpe if data_sharpe is not None else -999
    data_graham = info.get("graham_gap_pct")
    data_graham = data_graham if data_graham is not None else 0
    data_status = STATUS_PRIORITY.get(status, 5)

    return f"""
    <article class="card status-{status}" data-change="{data_change}" data-volume="{data_volume}"
      data-momentum="{data_momentum:.4f}" data-sharpe="{data_sharpe:.4f}" data-graham="{data_graham:.2f}"
      data-status="{data_status}">
      <div class="card-top">
        <div class="card-top-left">
          <label class="pick-label">
            <input type="checkbox" class="pick-checkbox" data-ticker="{info['ticker']}"
              data-name="{info['name']}" data-price="{info['price']}">
          </label>
          <div class="name">{info['name']}<span class="ticker">{info['ticker']}</span></div>
        </div>
        <span class="chip chip-{status}">{label}</span>
      </div>
      <div class="price-row">
        <span class="price">{fmt_price(info['price'])}<span class="yen">円</span></span>
        {change_html}
      </div>
      {render_tradingview_widget(info['ticker'])}
      {render_sparkline(info)}
      {detail_html}
    </article>
    """


def render_criteria_legend():
    """買い/売りの判断基準を数値つきでページ上部に明示する"""
    return f"""
    <section class="criteria-zone">
      <h2 class="section-title">📐 買い・売りの判断基準</h2>
      <div class="criteria-card">
        <div class="criteria-row">
          <span class="chip chip-buy">買い</span>
          <span>ゴールデンクロス発生から{config.GOLDEN_CROSS_RECENT_DAYS}営業日以内 かつ
            出来高が直近{config.VOLUME_LOOKBACK_WINDOW}日平均の{config.VOLUME_MULTIPLIER}倍以上 かつ
            売上成長率が{config.FUNDAMENTAL_MIN_REVENUE_GROWTH * 100:.0f}%を上回っている</span>
        </div>
        <div class="criteria-row">
          <span class="chip chip-sell">売り</span>
          <span>デッドクロス発生 または 購入価格から{config.STOP_LOSS_PCT * 100:.0f}%下落（損切り） または
            購入価格から+{config.TAKE_PROFIT_PCT * 100:.0f}%上昇（利確） のいずれか</span>
        </div>
        <div class="criteria-row">
          <span class="chip chip-watch">監視中</span>
          <span>上記の買い条件を満たしていない銘柄。「勢いスコア」で有望度を参考表示</span>
        </div>
      </div>
      <p class="fee-note">💰 {config.FEE_NOTE}</p>
    </section>
    """


def render_sort_bar(target_id):
    return f"""
    <div class="sort-bar" data-target="{target_id}" role="group" aria-label="並び替え">
      <button type="button" class="sort-btn is-active" data-sort="status">既定</button>
      <button type="button" class="sort-btn" data-sort="change">値動き</button>
      <button type="button" class="sort-btn" data-sort="volume">出来高</button>
      <button type="button" class="sort-btn" data-sort="momentum">勢い</button>
      <button type="button" class="sort-btn" data-sort="sharpe">シャープ</button>
      <button type="button" class="sort-btn" data-sort="graham">割安度</button>
    </div>
    """


def render_market_section(market_finds, scan_meta):
    if market_finds:
        sorted_finds = sorted(market_finds, key=lambda r: r.get("momentum_score", 0), reverse=True)
        cards = "".join(render_card(r) for r in sorted_finds)
        meta_txt = ""
        if scan_meta and scan_meta.get("scanned_at"):
            meta_txt = (
                f'<p class="scan-meta">東証上場{scan_meta.get("universe_size", "?")}銘柄をスキャンし'
                f'{scan_meta.get("total_found", "?")}件が条件に合致。勢いスコア上位{len(sorted_finds)}件を表示'
                f' ／ スキャン日時: {scan_meta["scanned_at"]}</p>'
            )
        return f"""
        <section class="market-scan-zone">
          <div class="section-header-row">
            <h2 class="section-title">🔍 全上場銘柄スキャンで見つかった買い候補</h2>
            {render_sort_bar("market-scan-cards")}
          </div>
          {meta_txt}
          <div class="card-list" id="market-scan-cards">{cards}</div>
        </section>
        """
    if scan_meta:
        return f"""
        <section class="market-scan-zone market-scan-empty">
          <p class="empty-msg">全銘柄スキャン（東証上場{scan_meta.get('universe_size', '?')}銘柄、
            {scan_meta.get('scanned_at', '-')}時点）では該当銘柄が見つかりませんでした。</p>
        </section>
        """
    return ""


def render_search_section(all_stocks):
    """東証上場の内国株式ほぼ全銘柄（ウォッチリスト外も含む）を対象にした検索窓。
    market_scan.py が保存した全銘柄の基本情報(all_stocks)をそのままページに埋め込み、
    JS側で銘柄名・コードによる絞り込みと、選んだ銘柄のリアルタイムチャート表示を行う。
    全銘柄分の詳細分析（ニュース・決算等）は重すぎるため行わず、基本情報+ライブチャートのみ。"""
    stocks_json = json.dumps(all_stocks, ensure_ascii=False)
    count = len(all_stocks)
    hint = f"（対象: 全{count:,}銘柄。市場が開いていない時間帯は前回スキャン時点のデータです）" if count else \
        "（まだ全銘柄スキャンのデータがありません。しばらくすると使えるようになります）"

    return f"""
    <section class="search-zone">
      <h2 class="section-title">🔎 全銘柄検索</h2>
      <p class="search-hint">ウォッチリストに入っていない銘柄も、銘柄名・銘柄コードで検索できます{hint}</p>
      <div class="search-box">
        <input type="text" id="stock-search-input" placeholder="例: トヨタ / 7203" autocomplete="off">
      </div>
      <div id="stock-search-results" class="search-results"></div>
      <div id="stock-search-detail" class="search-detail"></div>
    </section>
    <script id="all-stocks-data" type="application/json">{stocks_json}</script>
    <script>
    (function() {{
      var ALL_STOCKS = JSON.parse(document.getElementById('all-stocks-data').textContent || '[]');
      var input = document.getElementById('stock-search-input');
      var resultsEl = document.getElementById('stock-search-results');
      var detailEl = document.getElementById('stock-search-detail');
      if (!input) return;

      function fmtPct(v) {{
        if (v === null || v === undefined) return '-';
        var sign = v >= 0 ? '+' : '';
        return sign + v.toFixed(2) + '%';
      }}

      function showDetail(stock) {{
        var code = stock.ticker.replace('.T', '');
        var widgetId = 'tv_search_' + code;
        var changeCls = (stock.change_pct || 0) >= 0 ? 'rise' : 'fall';
        detailEl.innerHTML =
          '<div class="search-detail-card">' +
          '<div class="card-top"><div class="name">' + stock.name +
          '<span class="ticker">' + stock.ticker + '</span></div></div>' +
          '<div class="price-row"><span class="price">' + stock.price.toLocaleString('ja-JP', {{minimumFractionDigits: 1, maximumFractionDigits: 1}}) +
          '<span class="yen">円</span></span>' +
          '<span class="change ' + changeCls + '">' + fmtPct(stock.change_pct) + '</span></div>' +
          '<div class="tv-widget-container" id="' + widgetId + '"></div>' +
          '</div>';

        var root = document.documentElement;
        var theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        var forced = root.getAttribute('data-theme');
        if (forced === 'dark' || forced === 'light') theme = forced;
        var el = document.getElementById(widgetId);
        var s = document.createElement('script');
        s.src = 'https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js';
        s.async = true;
        s.text = JSON.stringify({{
          symbol: 'TSE:' + code,
          width: '100%',
          height: 220,
          locale: 'ja',
          dateRange: '1D',
          colorTheme: theme,
          isTransparent: true,
          autosize: true
        }});
        el.appendChild(s);
      }}

      function render(query) {{
        var q = query.trim().toLowerCase();
        if (!q) {{
          resultsEl.innerHTML = '';
          resultsEl.classList.remove('is-open');
          return;
        }}
        var matches = ALL_STOCKS.filter(function(s) {{
          var code = s.ticker.replace('.T', '');
          return s.name.toLowerCase().indexOf(q) !== -1 || code.indexOf(q) !== -1;
        }}).slice(0, 20);

        if (matches.length === 0) {{
          resultsEl.innerHTML = '<div class="search-empty">該当する銘柄が見つかりません</div>';
          resultsEl.classList.add('is-open');
          return;
        }}

        resultsEl.innerHTML = matches.map(function(s) {{
          var changeCls = (s.change_pct || 0) >= 0 ? 'rise' : 'fall';
          return '<button type="button" class="search-result-row" data-ticker="' + s.ticker + '">' +
            '<span class="search-result-name">' + s.name + '<span class="ticker">' + s.ticker + '</span></span>' +
            '<span class="search-result-price">' + s.price.toLocaleString('ja-JP') + '円 ' +
            '<span class="change ' + changeCls + '">' + fmtPct(s.change_pct) + '</span></span>' +
            '</button>';
        }}).join('');
        resultsEl.classList.add('is-open');

        resultsEl.querySelectorAll('.search-result-row').forEach(function(btn) {{
          btn.addEventListener('click', function() {{
            var stock = ALL_STOCKS.find(function(s) {{ return s.ticker === btn.getAttribute('data-ticker'); }});
            if (stock) showDetail(stock);
            resultsEl.classList.remove('is-open');
            input.value = stock ? stock.name : '';
          }});
        }});
      }}

      input.addEventListener('input', function() {{ render(input.value); }});
      input.addEventListener('focus', function() {{ if (input.value.trim()) render(input.value); }});
      document.addEventListener('click', function(e) {{
        if (!resultsEl.contains(e.target) && e.target !== input) {{
          resultsEl.classList.remove('is-open');
        }}
      }});
    }})();
    </script>
    """


def build_candidate_json(results, market_finds):
    """ポートフォリオ計算機がJS側で使う候補データをまとめる（保有中・エラーは除く）。
    シャープレシオが低すぎる・利確までの期間目安が長すぎる（または算出不可）銘柄は、
    予算内であっても品質ゲートで最初から除外する。"""
    seen = set()
    candidates = []
    for r in list(results) + list(market_finds):
        if r.get("status") == "error" or r.get("holding"):
            continue
        t = r["ticker"]
        if t in seen:
            continue
        seen.add(t)

        sharpe = r.get("sharpe")
        days_to_profit = r.get("days_to_profit")
        if sharpe is None or sharpe < config.MIN_SHARPE_FOR_PORTFOLIO:
            continue
        if days_to_profit is None or days_to_profit > config.MAX_DAYS_TO_PROFIT_FOR_PORTFOLIO:
            continue

        profile = r.get("profile") or {}
        candidates.append({
            "ticker": t,
            "name": r["name"],
            "price": r["price"],
            "momentum": round(r.get("momentum_score", 0) or 0, 4),
            "sharpe": round(sharpe, 2),
            "volatility": round(r.get("volatility") or 0, 4),
            "daysToProfit": round(days_to_profit, 1),
            "revenueGrowth": profile.get("revenue_growth"),
            "grahamGapPct": r.get("graham_gap_pct"),
            "sector": profile.get("sector"),
            "volumeRatio": r.get("volume_ratio"),
        })
    candidates.sort(key=lambda c: c["momentum"], reverse=True)
    return json.dumps(candidates, ensure_ascii=False)


def render_html(results, market_finds=None, scan_meta=None, all_stocks=None):
    market_finds = market_finds or []
    all_stocks = all_stocks or []
    market_section = render_market_section(market_finds, scan_meta)
    search_section = render_search_section(all_stocks)
    results_sorted = sorted(results, key=lambda r: STATUS_PRIORITY.get(r["status"], 5))
    action_items = [r for r in results_sorted if r["status"] in ("buy", "sell")]
    candidate_json = build_candidate_json(results_sorted, market_finds)

    action_section = ""
    if action_items:
        cards = "".join(render_card(r) for r in action_items)
        action_section = f"""
        <section class="action-zone">
          <h2 class="section-title">今すぐ確認</h2>
          <div class="card-list">{cards}</div>
        </section>
        """
    else:
        action_section = """
        <section class="action-zone action-zone-empty">
          <p class="empty-msg">本日、緊急の売買シグナルはありません。</p>
        </section>
        """

    pickup_section = render_pickup(results_sorted)
    all_cards = "".join(render_card(r) for r in results_sorted)
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<title>投資部ダッシュボード</title>
<style>
:root {{
  --ink: #12161c;
  --paper: #f1f4f7;
  --surface: #ffffff;
  --surface-2: #e7ebef;
  --text-primary: #12161c;
  --text-secondary: #4b5563;
  --text-muted: #7c8592;
  --border: #d8dee5;
  --accent: #2e6b65;
  --rise: #b8393b;
  --fall: #2f6fa6;
  --buy: #b8842c;
  --sell: #93395a;
}}

@media (prefers-color-scheme: dark) {{
  :root {{
    --paper: #10151b;
    --surface: #171d25;
    --surface-2: #1f2731;
    --text-primary: #eef1f5;
    --text-secondary: #b7c0cc;
    --text-muted: #7d8794;
    --border: #2a323d;
    --accent: #6fbdb3;
    --rise: #e0696b;
    --fall: #6fa8d6;
    --buy: #e0ac54;
    --sell: #d4789a;
  }}
}}

:root[data-theme="dark"] {{
  --paper: #10151b;
  --surface: #171d25;
  --surface-2: #1f2731;
  --text-primary: #eef1f5;
  --text-secondary: #b7c0cc;
  --text-muted: #7d8794;
  --border: #2a323d;
  --accent: #6fbdb3;
  --rise: #e0696b;
  --fall: #6fa8d6;
  --buy: #e0ac54;
  --sell: #d4789a;
}}

:root[data-theme="light"] {{
  --paper: #f1f4f7;
  --surface: #ffffff;
  --surface-2: #e7ebef;
  --text-primary: #12161c;
  --text-secondary: #4b5563;
  --text-muted: #7c8592;
  --border: #d8dee5;
  --accent: #2e6b65;
  --rise: #b8393b;
  --fall: #2f6fa6;
  --buy: #b8842c;
  --sell: #93395a;
}}

* {{ box-sizing: border-box; }}

html, body {{
  height: 100%;
  margin: 0;
  overflow: hidden;
  background: var(--paper);
  color: var(--text-primary);
  font-family: "Hiragino Sans", "Yu Gothic Medium", YuGothic, "Noto Sans JP",
    system-ui, -apple-system, sans-serif;
}}

/* iOSのWebView内で「回転しないとスクロールできない」不具合の対策:
   bodyのスクロールに任せず、position:fixedの専用スクロールコンテナに任せる */
#scroll-root {{
  position: fixed;
  inset: 0;
  overflow-y: auto;
  overflow-x: hidden;
  -webkit-overflow-scrolling: touch;
}}

.page {{
  max-width: 640px;
  margin: 0 auto;
  padding: 20px 16px 48px;
}}

#tap-top {{
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 24px;
  z-index: 999;
  cursor: pointer;
}}

header.app-header {{
  padding-bottom: 14px;
  margin-bottom: 18px;
  border-bottom: 2px solid var(--accent);
}}

.app-header .title-row {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}}

.app-header h1 {{
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  margin: 0;
  text-wrap: balance;
}}

.app-header .updated {{
  font-size: 0.72rem;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}

.app-header .subtitle {{
  margin: 4px 0 0;
  font-size: 0.8rem;
  color: var(--text-secondary);
}}

.budget-bar {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 12px;
  padding: 10px 12px;
  background: var(--surface);
  border: 1px solid var(--accent);
  border-radius: 10px;
}}

.budget-bar label {{
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--accent);
}}

.budget-bar input {{
  flex: 1;
  min-width: 0;
  font-size: 1rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--paper);
  color: var(--text-primary);
}}

.budget-bar input:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}}

.budget-bar-hint {{
  margin: 6px 0 0;
  font-size: 0.66rem;
  color: var(--text-muted);
}}

.view-tabs {{
  display: flex;
  gap: 6px;
  margin: 14px 0;
}}

.view-tab-btn {{
  flex: 1;
  font-family: inherit;
  font-size: 0.76rem;
  font-weight: 700;
  color: var(--text-secondary);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 10px;
  cursor: pointer;
}}

.view-tab-btn.is-active {{
  background: var(--accent);
  color: var(--paper);
  border-color: var(--accent);
}}

.view-tab-btn:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}}

.picked-count {{
  display: inline-block;
  margin-left: 4px;
  font-variant-numeric: tabular-nums;
}}

.selected-zone {{
  margin: 18px 0;
}}

.selected-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 10px;
}}

.selected-card-top {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 8px;
}}

.selected-remove-btn {{
  font-family: inherit;
  font-size: 0.66rem;
  color: var(--text-muted);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 2px 8px;
  cursor: pointer;
  flex-shrink: 0;
}}

.growth-plan-zone {{
  margin: 18px 0;
}}

.growth-plan-card {{
  margin-top: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}}

.growth-plan-inputs {{
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}}

.growth-plan-inputs label {{
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.72rem;
  color: var(--text-muted);
  flex: 1;
  min-width: 120px;
}}

.growth-plan-inputs input {{
  font-size: 0.9rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  padding: 6px 8px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--paper);
  color: var(--text-primary);
}}

.growth-plan-summary {{
  margin: 8px 0 0;
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--accent);
}}

.growth-plan-feasibility {{
  margin-top: 8px;
  font-size: 0.76rem;
  color: var(--text-secondary);
  line-height: 1.7;
}}

.growth-plan-feasibility p {{
  margin: 0 0 4px;
}}

.feasibility-tag {{
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  margin-bottom: 4px;
}}

.feasibility-ok {{ background: color-mix(in srgb, var(--buy) 20%, var(--surface)); color: var(--buy); }}
.feasibility-hard {{ background: var(--surface-2); color: var(--text-secondary); }}
.feasibility-unrealistic {{ background: color-mix(in srgb, var(--sell) 20%, var(--surface)); color: var(--sell); }}

.growth-plan-note {{
  margin: 6px 0 0;
  font-size: 0.66rem;
  color: var(--text-muted);
  line-height: 1.6;
}}

.criteria-zone {{
  margin: 18px 0;
}}

.criteria-card {{
  margin-top: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}}

.criteria-row {{
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: 0.76rem;
  color: var(--text-secondary);
  line-height: 1.6;
}}

.criteria-row .chip {{
  flex-shrink: 0;
  margin-top: 1px;
}}

.fee-note {{
  margin: 8px 0 0;
  font-size: 0.7rem;
  color: var(--text-muted);
  line-height: 1.6;
}}

.portfolio-zone {{
  margin: 18px 0 22px;
}}

.portfolio-card {{
  margin-top: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 14px;
}}

.portfolio-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  padding: 6px 0;
  border-top: 1px solid var(--border);
  font-size: 0.8rem;
}}

.portfolio-row:first-child {{ border-top: none; }}

.portfolio-name {{
  font-weight: 700;
}}

.portfolio-detail {{
  font-variant-numeric: tabular-nums;
  color: var(--text-secondary);
  white-space: nowrap;
}}

.portfolio-total {{
  font-weight: 700;
  border-top: 2px solid var(--accent);
}}

.portfolio-reason {{
  font-size: 0.68rem;
  color: var(--text-muted);
  padding: 0 0 4px;
  border-top: none;
}}

.portfolio-reason-label {{
  display: inline-block;
  min-width: 4.5em;
  font-weight: 700;
  color: var(--accent);
  margin-right: 4px;
}}

.quadrant-tag {{
  display: inline-block;
  margin-left: 6px;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-secondary);
  vertical-align: middle;
}}

.portfolio-excluded {{
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--border);
}}

.portfolio-excluded-title {{
  margin: 0 0 6px;
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--text-muted);
}}

.portfolio-excluded-row {{
  display: flex;
  flex-direction: column;
  gap: 1px;
  font-size: 0.68rem;
  color: var(--text-muted);
  padding: 4px 0;
}}

.section-title {{
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  text-transform: uppercase;
  margin: 0 0 10px;
}}

.market-scan-zone {{
  margin-bottom: 22px;
}}

.market-scan-zone .card-list {{
  margin-top: 10px;
}}

.scan-meta {{
  margin: 0;
  font-size: 0.68rem;
  color: var(--text-muted);
}}

.market-scan-empty {{
  padding: 14px;
  border: 1px dashed var(--border);
  border-radius: 10px;
  text-align: center;
}}

.pickup-zone {{
  margin-bottom: 22px;
}}

.pickup-card {{
  background: color-mix(in srgb, var(--accent) 10%, var(--surface));
  border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--border));
  border-radius: 10px;
  padding: 14px;
}}

.pickup-heading {{
  margin: 0 0 8px;
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--accent);
}}

.pickup-row {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  padding: 6px 0;
  border-top: 1px solid var(--border);
}}

.pickup-row:first-of-type {{ border-top: none; }}

.pickup-name {{
  font-weight: 700;
  font-size: 0.88rem;
}}

.pickup-fit {{
  font-size: 0.76rem;
  color: var(--text-secondary);
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}}

.budget-line {{
  margin-top: 6px;
  font-size: 0.72rem;
  color: var(--text-muted);
}}

.more-details {{
  margin-top: 8px;
}}

.more-details summary {{
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--accent);
  cursor: pointer;
  list-style: none;
}}

.more-details summary::-webkit-details-marker {{
  display: none;
}}

.more-details summary::before {{
  content: "▸ ";
}}

.more-details[open] summary::before {{
  content: "▾ ";
}}

.more-details[open] summary {{
  margin-bottom: 6px;
}}

.profile-facts {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  font-size: 0.72rem;
  color: var(--text-secondary);
  margin-bottom: 6px;
}}

.profile-summary {{
  margin: 0;
  font-size: 0.76rem;
  color: var(--text-secondary);
  line-height: 1.7;
}}

.profile-note {{
  font-size: 0.66rem;
  color: var(--text-muted);
}}

.valuation-tag {{
  display: inline-block;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 999px;
  margin-left: 2px;
}}

.tag-cheap {{ background: color-mix(in srgb, var(--buy) 20%, var(--surface)); color: var(--buy); }}
.tag-normal {{ background: var(--surface-2); color: var(--text-secondary); }}
.tag-expensive {{ background: color-mix(in srgb, var(--sell) 20%, var(--surface)); color: var(--sell); }}
.tag-muted {{ background: var(--surface-2); color: var(--text-muted); }}

.profile-summary-muted {{
  color: var(--text-muted);
  font-style: italic;
}}

.financials-title {{
  margin: 10px 0 4px;
  font-size: 0.7rem;
  font-weight: 700;
  color: var(--text-muted);
}}

.financials-table-wrap {{
  overflow-x: auto;
}}

.financials-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.72rem;
  font-variant-numeric: tabular-nums;
}}

.financials-table th, .financials-table td {{
  text-align: right;
  padding: 4px 6px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}

.financials-table th:first-child, .financials-table td:first-child {{
  text-align: left;
}}

.financials-table thead th {{
  color: var(--text-muted);
  font-weight: 600;
  font-size: 0.66rem;
}}

.financials-unit {{
  margin: 4px 0 0;
  font-size: 0.62rem;
  color: var(--text-muted);
}}

.financials-empty {{
  margin: 0;
  font-size: 0.72rem;
  color: var(--text-muted);
}}

.budget {{
  margin-top: 6px;
  font-size: 0.72rem;
  padding: 4px 8px;
  border-radius: 6px;
  display: inline-block;
}}

.budget-ok {{
  background: color-mix(in srgb, var(--buy) 16%, var(--surface));
  color: var(--buy);
}}

.budget-skabu {{
  background: var(--surface-2);
  color: var(--text-secondary);
}}

.budget-over {{
  background: var(--surface-2);
  color: var(--text-muted);
}}

.action-zone {{
  margin-bottom: 26px;
}}

.action-zone-empty {{
  padding: 16px;
  border: 1px dashed var(--border);
  border-radius: 10px;
  text-align: center;
}}

.empty-msg {{
  margin: 0;
  font-size: 0.85rem;
  color: var(--text-secondary);
}}

.card-list {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
}}

.card.status-buy {{ border-color: color-mix(in srgb, var(--buy) 55%, var(--border)); }}
.card.status-sell {{ border-color: color-mix(in srgb, var(--sell) 55%, var(--border)); }}

.card-top {{
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}}

.card-top-left {{
  display: flex;
  align-items: flex-start;
  gap: 8px;
  min-width: 0;
}}

.pick-label {{
  flex-shrink: 0;
  padding-top: 2px;
}}

.pick-checkbox {{
  width: 18px;
  height: 18px;
  cursor: pointer;
  accent-color: var(--accent);
}}

.name {{
  font-weight: 700;
  font-size: 0.95rem;
}}

.ticker {{
  display: inline-block;
  margin-left: 6px;
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  font-size: 0.72rem;
  color: var(--text-muted);
  font-weight: 400;
}}

.chip {{
  flex-shrink: 0;
  font-size: 0.68rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 999px;
  white-space: nowrap;
  letter-spacing: 0.02em;
}}

.chip-buy {{ background: color-mix(in srgb, var(--buy) 22%, var(--surface)); color: var(--buy); }}
.chip-sell {{ background: color-mix(in srgb, var(--sell) 22%, var(--surface)); color: var(--sell); }}
.chip-hold {{ background: var(--surface-2); color: var(--text-secondary); }}
.chip-watch {{ background: var(--surface-2); color: var(--text-muted); }}
.chip-error {{ background: var(--surface-2); color: var(--text-muted); }}

.price-row {{
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-top: 8px;
}}

.price {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 1.3rem;
  font-weight: 700;
}}

.yen {{
  font-size: 0.7rem;
  font-weight: 400;
  color: var(--text-muted);
  margin-left: 2px;
}}

.change {{
  font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 0.8rem;
  font-weight: 600;
}}

.rise {{ color: var(--rise); }}
.fall {{ color: var(--fall); }}

.detail {{
  margin-top: 6px;
  font-size: 0.76rem;
  color: var(--text-secondary);
}}

.dot-sep {{ margin: 0 4px; color: var(--text-muted); }}

.tags {{
  margin-top: 6px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}

.tag {{
  font-size: 0.68rem;
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--accent);
}}

.reasons {{
  margin-top: 6px;
  font-size: 0.76rem;
  color: var(--sell);
  font-weight: 600;
}}

.error-msg {{
  margin: 6px 0 0;
  font-size: 0.8rem;
  color: var(--text-muted);
}}

.gauge {{
  margin-top: 10px;
}}

.gauge-track {{
  position: relative;
  height: 6px;
  border-radius: 999px;
  background: linear-gradient(to right, var(--fall), var(--surface-2) 50%, var(--rise));
}}

.gauge-marker {{
  position: absolute;
  top: 50%;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--text-primary);
  border: 2px solid var(--surface);
  transform: translate(-50%, -50%);
}}

.gauge-labels {{
  display: flex;
  justify-content: space-between;
  margin-top: 5px;
  font-size: 0.66rem;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}}

.gauge-labels .pnl {{ font-weight: 700; }}

.profit-estimate {{
  margin-top: 8px;
  padding: 8px 10px;
  background: var(--surface-2);
  border-radius: 8px;
}}

.profit-estimate-title {{
  font-size: 0.68rem;
  color: var(--text-muted);
  margin-bottom: 4px;
}}

.profit-estimate-row {{
  display: flex;
  gap: 14px;
  font-size: 0.78rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}}

.pnl-yen {{
  font-weight: 400;
  font-size: 0.9em;
}}

.pickup-profit {{
  font-size: 0.72rem;
  font-variant-numeric: tabular-nums;
}}

.glossary {{
  margin-top: 26px;
}}

.glossary-list {{
  display: flex;
  flex-direction: column;
  gap: 10px;
}}

.glossary-row {{
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}}

.glossary-row:last-child {{ border-bottom: none; }}

.glossary-term {{
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--accent);
}}

.glossary-desc {{
  font-size: 0.74rem;
  color: var(--text-secondary);
  line-height: 1.6;
}}

.section-header-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}}

.section-header-row .section-title {{ margin: 0; }}

.sort-bar {{
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}}

.sort-btn {{
  font-family: inherit;
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-secondary);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 4px 10px;
  cursor: pointer;
}}

.sort-btn.is-active {{
  background: var(--accent);
  color: var(--paper);
  border-color: var(--accent);
}}

.sort-btn:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}}

.tv-widget-container {{
  margin-top: 10px;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}}

.search-zone {{
  margin-bottom: 18px;
  position: relative;
}}

.search-hint {{
  font-size: 0.72rem;
  color: var(--text-muted);
  margin: 4px 0 8px;
}}

.search-box input {{
  width: 100%;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 0.9rem;
}}

.search-box input:focus {{
  outline: 2px solid var(--accent);
  outline-offset: 1px;
}}

.search-results {{
  display: none;
  position: relative;
  z-index: 20;
  margin-top: 4px;
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
}}

.search-results.is-open {{
  display: block;
}}

.search-empty {{
  padding: 12px;
  font-size: 0.8rem;
  color: var(--text-muted);
}}

.search-result-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 10px 12px;
  border: none;
  border-bottom: 1px solid var(--border);
  background: transparent;
  color: var(--text-primary);
  font-size: 0.82rem;
  text-align: left;
  cursor: pointer;
}}

.search-result-row:last-child {{
  border-bottom: none;
}}

.search-result-row:hover, .search-result-row:focus-visible {{
  background: var(--surface-2);
}}

.search-result-name {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}

.search-result-price {{
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}}

.search-detail {{
  margin-top: 10px;
}}

.search-detail-card {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px;
  background: var(--surface);
}}

.sparkline {{
  margin-top: 10px;
}}

.sparkline svg {{
  display: block;
}}

.spark-label {{
  font-size: 7px;
  fill: var(--text-muted);
}}

.sparkline-caption {{
  display: flex;
  justify-content: space-between;
  margin-top: 2px;
  font-size: 0.62rem;
  color: var(--text-muted);
}}

.metric-badge {{
  display: inline-block;
  margin-top: 2px;
  font-size: 0.74rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
  font-variant-numeric: tabular-nums;
}}

.badge-good {{ background: color-mix(in srgb, var(--rise) 18%, var(--surface)); color: var(--rise); }}
.badge-bad {{ background: color-mix(in srgb, var(--fall) 18%, var(--surface)); color: var(--fall); }}
.badge-neutral {{ background: var(--surface-2); color: var(--text-secondary); }}
.badge-muted {{ background: var(--surface-2); color: var(--text-muted); }}

.graham-gauge {{
  margin-top: 8px;
}}

.graham-track {{
  background: linear-gradient(to right, var(--buy), var(--surface-2) 50%, var(--sell));
}}

.quant {{
  margin-top: 10px;
  padding: 8px 10px;
  background: var(--surface-2);
  border-radius: 8px;
}}

.quant-row {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  font-size: 0.76rem;
  font-variant-numeric: tabular-nums;
}}

.quant-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 8px;
}}

.quant-grid > div {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}

.pro-fund-grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin: 6px 0 8px;
}}

.pro-fund-grid > div {{
  display: flex;
  flex-direction: column;
  gap: 2px;
}}

.news-list {{
  margin: 4px 0 0;
  padding-left: 18px;
  font-size: 0.74rem;
  color: var(--text-secondary);
  line-height: 1.7;
}}

.news-list li {{
  margin-bottom: 8px;
}}

.news-list a {{
  color: var(--accent);
  text-decoration: underline;
}}

.news-impact {{
  margin-top: 2px;
  font-size: 0.68rem;
  color: var(--text-muted);
}}

.quant-value {{
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  font-size: 0.82rem;
}}

.quant-label {{
  font-size: 0.62rem;
  color: var(--text-muted);
  white-space: nowrap;
}}

.narrative {{
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}}

.narrative-row {{
  display: flex;
  gap: 6px;
  font-size: 0.76rem;
  color: var(--text-secondary);
  line-height: 1.6;
}}

.narrative-label {{
  flex-shrink: 0;
  font-size: 0.66rem;
  font-weight: 700;
  color: var(--text-muted);
  padding-top: 0.1em;
  white-space: nowrap;
}}

.narrative-basis {{
  color: var(--text-muted);
  font-size: 0.72rem;
}}

.timing {{
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}}

.timing-row {{
  display: flex;
  gap: 6px;
  font-size: 0.76rem;
  line-height: 1.6;
}}

.timing-label {{
  flex-shrink: 0;
  font-size: 0.66rem;
  font-weight: 700;
  color: var(--accent);
  padding-top: 0.1em;
  white-space: nowrap;
}}

footer.disclaimer {{
  margin-top: 28px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  font-size: 0.7rem;
  color: var(--text-muted);
  line-height: 1.6;
}}
</style>
<div id="scroll-root">
<div class="page">
  <header class="app-header">
    <div class="title-row">
      <h1>投資部ダッシュボード</h1>
      <span class="updated">最終更新 {updated_at}</span>
    </div>
    <p class="subtitle">ゴールデンクロス + 出来高 + ファンダメンタルで個別株の売買タイミングを監視</p>
    <div class="budget-bar">
      <label for="budget-input">予算</label>
      <input type="number" id="budget-input" value="{config.BUDGET_JPY}" step="1000" min="0" inputmode="numeric">
      <span>円</span>
    </div>
    <p class="budget-bar-hint">↑ここを変えると、全カードの購入可否・利益目安・下のポートフォリオ計算が連動します</p>
  </header>

  {search_section}

  <div class="view-tabs" role="tablist" aria-label="表示切り替え">
    <button type="button" class="view-tab-btn is-active" data-view="all">全銘柄</button>
    <button type="button" class="view-tab-btn" data-view="selected">✅ 選んだ銘柄でポートフォリオ<span id="picked-count" class="picked-count"></span></button>
  </div>

  <div id="view-all">

  <section class="growth-plan-zone">
    <h2 class="section-title">🎯 目指す収益の計画</h2>
    <div class="growth-plan-card">
      <div class="growth-plan-inputs">
        <label>目標金額
          <input type="number" id="target-amount-input" value="{config.BUDGET_JPY * 3}" step="1000" min="0" inputmode="numeric">円
        </label>
        <label>達成したい日数
          <input type="number" id="target-days-input" value="60" step="1" min="1" inputmode="numeric">営業日
        </label>
      </div>
      <div id="growth-plan-chart"></div>
      <p id="growth-plan-summary" class="growth-plan-summary"></p>
      <div id="growth-plan-feasibility" class="growth-plan-feasibility"></div>
      <div id="growth-plan-achieve" class="growth-plan-feasibility"></div>
      <p class="growth-plan-note">これは「目標に届くには何%ペースの成長が必要か」を単純な複利計算で可視化したものです。
        実際に到達する保証はありません。上の予算・銘柄の利確目安と見比べて、現実的かどうかの参考にしてください。</p>
    </div>
  </section>

  {render_criteria_legend()}

  <section class="portfolio-zone">
    <h2 class="section-title">🧮 おすすめポートフォリオ計算</h2>
    <p class="scan-meta">シャープレシオ{config.MIN_SHARPE_FOR_PORTFOLIO:.1f}以上・利確までの目安{config.MAX_DAYS_TO_PROFIT_FOR_PORTFOLIO}営業日以内の銘柄だけを候補にし、
      「短期×がっつり」型を最優先しつつ、複数のタイプを組み合わせて100株単位で機械的に配分した一例です。</p>
    <div class="portfolio-card" id="portfolio-result"></div>
  </section>

  {market_section}

  {pickup_section}

  {action_section}

  <section class="watchlist">
    <div class="section-header-row">
      <h2 class="section-title">ウォッチリスト全体</h2>
      {render_sort_bar("watchlist-cards")}
    </div>
    <div class="card-list" id="watchlist-cards">
      {all_cards}
    </div>
  </section>

  </div>

  <div id="view-selected" style="display:none">
    <section class="selected-zone">
      <h2 class="section-title">✅ 選んだ銘柄でポートフォリオ</h2>
      <p class="scan-meta">各カードのチェックボックスで選んだ銘柄が表示されます。チェックした瞬間の株価を基準に、
        損切り・利確ラインと売るタイミングを計算します（この情報はこの端末のブラウザにのみ保存されます）。</p>
      <div id="selected-portfolio-result"></div>
    </section>
  </div>

  {render_glossary()}

  <footer class="disclaimer">
    これは投資助言ではありません。表示内容は過去データに基づく機械的な判定であり、将来の値動きを保証するものではありません。
    「うまくいけば／しくじれば」の金額も、あらかじめ決めたルール（損切り-8%・利確+15%）に達した場合の単純計算であり、
    実際にその価格で売買できる保証はありません。発注は自動化されておらず、最終判断と発注操作はご本人が行ってください。
  </footer>
</div>
</div>
<div id="tap-top" aria-hidden="true"></div>
<script>
(function () {{
  // 画面いちばん上をタップ/クリックしたら、スクロールコンテナの先頭へ戻る
  // （bodyではなく#scroll-rootがスクロールを担当しているため、OS標準の「ステータスバーをタップで最上部へ」は効かない代わりにこれで再現する）
  var tapTop = document.getElementById('tap-top');
  var scrollRoot = document.getElementById('scroll-root');
  if (tapTop && scrollRoot) {{
    tapTop.addEventListener('click', function () {{
      scrollRoot.scrollTo({{ top: 0, behavior: 'smooth' }});
    }});
  }}
}})();
(function () {{
  // ページ内の並び替えバーはそれぞれ data-target で自分の対象カードリストを持つ（複数セクションに対応）
  document.querySelectorAll('.sort-bar').forEach(function (bar) {{
    var container = document.getElementById(bar.getAttribute('data-target'));
    if (!container) return;
    var buttons = bar.querySelectorAll('.sort-btn');
    buttons.forEach(function (btn) {{
      btn.addEventListener('click', function () {{
        buttons.forEach(function (b) {{ b.classList.remove('is-active'); }});
        btn.classList.add('is-active');
        var key = btn.getAttribute('data-sort');
        var cards = Array.prototype.slice.call(container.children);
        cards.sort(function (a, b) {{
          if (key === 'status') {{
            return parseFloat(a.dataset.status) - parseFloat(b.dataset.status);
          }}
          if (key === 'change' || key === 'graham') {{
            return Math.abs(parseFloat(b.dataset[key])) - Math.abs(parseFloat(a.dataset[key]));
          }}
          return parseFloat(b.dataset[key]) - parseFloat(a.dataset[key]);
        }});
        cards.forEach(function (card) {{ container.appendChild(card); }});
      }});
    }});
  }});
}})();
</script>
<script type="application/json" id="candidate-data">{candidate_json}</script>
<script>
(function () {{
  var TAKE_PROFIT_PCT = {config.TAKE_PROFIT_PCT};
  var STOP_LOSS_PCT = {config.STOP_LOSS_PCT};
  var SHORT_TERM_DAYS = {config.SHORT_TERM_DAYS};
  var AGGRESSIVE_VOL_THRESHOLD = {config.AGGRESSIVE_VOL_THRESHOLD};
  var budgetInput = document.getElementById('budget-input');
  var portfolioResult = document.getElementById('portfolio-result');
  var candidateData = [];
  try {{
    candidateData = JSON.parse(document.getElementById('candidate-data').textContent);
  }} catch (e) {{ candidateData = []; }}

  function yen(n) {{
    return Math.round(n).toLocaleString('ja-JP');
  }}

  function recalcBudgetBoxes(budget) {{
    var boxes = document.querySelectorAll('.budget-box');
    boxes.forEach(function (box) {{
      var price = parseFloat(box.dataset.price);
      if (!price) return;
      var unitCost = price * 100;
      var fits = unitCost <= budget;
      var dtpRaw = box.dataset.daysToProfit;
      var periodTxt = (dtpRaw && dtpRaw !== '') ? ('約' + Math.round(parseFloat(dtpRaw)) + '営業日で') : '期間は現時点で見通せませんが、';
      var html = '<div class="budget-line">単元株数: 100株（' + yen(unitCost) + '円）</div>';
      if (fits) {{
        var profit = unitCost * TAKE_PROFIT_PCT;
        var loss = unitCost * STOP_LOSS_PCT;
        html += '<div class="budget budget-ok">予算' + yen(budget) + '円以内で購入可</div>';
        html += '<div class="profit-estimate">' +
          '<div class="profit-estimate-title">' + yen(unitCost) + '円分買った場合の目安（' + periodTxt + '利確ラインに届く想定）</div>' +
          '<div class="profit-estimate-row">' +
          '<span class="rise">うまくいけば +' + yen(profit) + '円</span>' +
          '<span class="fall">しくじれば ' + yen(loss) + '円</span>' +
          '</div></div>';
      }} else {{
        html += '<div class="budget budget-over">予算オーバー（あと' + yen(unitCost - budget) + '円足りません）</div>';
      }}
      box.innerHTML = html;
    }});
  }}

  function classify(c) {{
    var isShort = c.daysToProfit !== null && c.daysToProfit <= SHORT_TERM_DAYS;
    var isAggressive = c.volatility >= AGGRESSIVE_VOL_THRESHOLD;
    var timeLabel = isShort ? '短期' : '中長期';
    var powerLabel = isAggressive ? 'がっつり' : '着実';
    // 優先順位: 短期×がっつり(0) を最優先、次に短期×着実/中長期×がっつり(1)、最後に中長期×着実(2)
    var priority = isShort && isAggressive ? 0 : (isShort || isAggressive ? 1 : 2);
    return {{ timeLabel: timeLabel, powerLabel: powerLabel, priority: priority }};
  }}

  function fundamentalNote(c) {{
    var parts = [];
    if (c.revenueGrowth !== null && c.revenueGrowth !== undefined) {{
      parts.push('売上成長率' + (c.revenueGrowth * 100).toFixed(1) + '%');
    }} else {{
      parts.push('売上成長率データなし');
    }}
    if (c.grahamGapPct !== null && c.grahamGapPct !== undefined) {{
      parts.push('理論株価比' + (c.grahamGapPct >= 0 ? '+' : '') + c.grahamGapPct.toFixed(0) + '%');
    }}
    if (c.sector) {{
      parts.push(c.sector);
    }}
    return parts.join('・');
  }}

  function technicalNote(c) {{
    var parts = ['勢いスコア' + c.momentum.toFixed(2)];
    if (c.volumeRatio !== null && c.volumeRatio !== undefined) {{
      parts.push('出来高' + c.volumeRatio.toFixed(2) + '倍');
    }}
    parts.push('シャープ' + c.sharpe.toFixed(2));
    parts.push('利確目安' + c.daysToProfit.toFixed(0) + '営業日');
    return parts.join('・');
  }}

  function computePortfolio(budget) {{
    // 短期×がっつりを最優先にした並び順で貪欲法配分する（品質ゲート済みの候補のみが対象）
    var order = candidateData.map(function (c, i) {{
      var cls = classify(c);
      return Object.assign({{}}, c, cls, {{ rank: i + 1 }});
    }});
    order.sort(function (a, b) {{
      if (a.priority !== b.priority) return a.priority - b.priority;
      return b.momentum - a.momentum;
    }});

    var remaining = budget;
    var allocations = [];
    var byTicker = {{}};
    var guard = 0;
    var progress = true;
    while (remaining > 0 && progress && guard < 500) {{
      progress = false;
      guard += 1;
      for (var i = 0; i < order.length; i++) {{
        var c = order[i];
        var lotCost = c.price * 100;
        if (lotCost > 0 && lotCost <= remaining) {{
          if (!byTicker[c.ticker]) {{
            byTicker[c.ticker] = Object.assign({{}}, c, {{ lots: 0 }});
            allocations.push(byTicker[c.ticker]);
          }}
          byTicker[c.ticker].lots += 1;
          remaining -= lotCost;
          progress = true;
        }}
      }}
    }}
    var excluded = order.filter(function (c) {{ return !byTicker[c.ticker]; }}).slice(0, 5);
    return {{ allocations: allocations, remaining: remaining, excluded: excluded }};
  }}

  function getActivePortfolioStats(budget) {{
    // 「選んだ銘柄」があればそちらを優先し、なければおすすめポートフォリオで代用する
    var picks = getPicks();
    var tickers = Object.keys(picks);
    if (tickers.length) {{
      var totalCost = 0, totalProfit = 0, weightedDaysSum = 0, count = 0;
      tickers.forEach(function (t) {{
        var c = candidateData.filter(function (x) {{ return x.ticker === t; }})[0];
        var price = c ? c.price : picks[t].price;
        var daysToProfit = c ? c.daysToProfit : null;
        if (!price || !daysToProfit || daysToProfit <= 0) return;
        var cost = price * 100;
        totalCost += cost;
        totalProfit += cost * TAKE_PROFIT_PCT;
        weightedDaysSum += daysToProfit * cost;
        count += 1;
      }});
      if (count > 0) {{
        return {{ source: 'selected', totalCost: totalCost, totalProfit: totalProfit, weightedDays: weightedDaysSum / totalCost, count: count }};
      }}
    }}
    var result = computePortfolio(budget);
    if (result.allocations.length) {{
      var tc = 0, tp = 0, wd = 0;
      result.allocations.forEach(function (a) {{
        var cost = a.lots * a.price * 100;
        tc += cost;
        tp += cost * TAKE_PROFIT_PCT;
        wd += a.daysToProfit * cost;
      }});
      return {{ source: 'recommended', totalCost: tc, totalProfit: tp, weightedDays: tc > 0 ? wd / tc : 0, count: result.allocations.length }};
    }}
    return null;
  }}

  function renderPortfolio() {{
    var budget = parseFloat(budgetInput.value) || 0;
    if (!candidateData.length) {{
      portfolioResult.innerHTML = '<p class="empty-msg">' +
        'シャープレシオ・利確までの期間の品質ゲートを満たす候補が現在ありません。</p>';
      return;
    }}
    var result = computePortfolio(budget);
    if (!result.allocations.length) {{
      portfolioResult.innerHTML = '<p class="empty-msg">この予算では、候補銘柄を100株単位で購入できません。</p>';
      return;
    }}
    var rows = '';
    var totalCost = 0;
    var totalProfit = 0;
    var weightedDaysSum = 0;
    result.allocations.forEach(function (a) {{
      var shares = a.lots * 100;
      var cost = a.lots * a.price * 100;
      var profit = cost * TAKE_PROFIT_PCT;
      totalCost += cost;
      totalProfit += profit;
      weightedDaysSum += a.daysToProfit * cost;
      rows += '<div class="portfolio-row">' +
        '<span class="portfolio-name">' + a.name + '<span class="ticker">' + a.ticker + '</span>' +
        '<span class="quadrant-tag">' + a.timeLabel + '×' + a.powerLabel + '</span></span>' +
        '<span class="portfolio-detail">' + shares.toLocaleString('ja-JP') + '株（' + yen(cost) + '円）</span>' +
        '</div>' +
        '<div class="portfolio-reason">' +
        '<span class="portfolio-reason-label">テクニカル</span>' + technicalNote(a) + '</div>' +
        '<div class="portfolio-reason">' +
        '<span class="portfolio-reason-label">ファンダ</span>' + fundamentalNote(a) + '</div>' +
        '<div class="portfolio-reason">' +
        '<span class="portfolio-reason-label">見込み</span>約' + a.daysToProfit.toFixed(0) + '営業日で+' + yen(profit) + '円の目安</div>';
    }});

    var weightedDays = totalCost > 0 ? weightedDaysSum / totalCost : 0;
    var chartHtml = '';
    if (totalCost > 0 && totalProfit > 0 && weightedDays > 0) {{
      var built = buildGrowthSvg(totalCost, totalCost + totalProfit, weightedDays, 'var(--rise)');
      chartHtml = '<div class="sparkline">' + built.svg +
        '<div class="sparkline-caption"><span>投資額 ' + yen(totalCost) + '円</span>' +
        '<span>約' + weightedDays.toFixed(0) + '営業日後 ' + yen(totalCost + totalProfit) + '円（全銘柄利確想定）</span></div></div>';
    }}

    rows = chartHtml + rows;
    rows += '<div class="portfolio-row portfolio-total">' +
      '<span>合計</span><span>' + yen(totalCost) + '円（残り現金 ' + yen(result.remaining) + '円）</span></div>';

    if (result.excluded.length) {{
      rows += '<div class="portfolio-excluded">' +
        '<p class="portfolio-excluded-title">検討したが見送った候補</p>';
      result.excluded.forEach(function (c) {{
        var lotCost = c.price * 100;
        rows += '<div class="portfolio-excluded-row">' +
          '<span>' + c.name + '<span class="ticker">' + c.ticker + '</span>（' + c.timeLabel + '×' + c.powerLabel + '）</span>' +
          '<span>1ロット' + yen(lotCost) + '円 &gt; 残り予算' + yen(result.remaining) + '円のため見送り</span>' +
          '</div>';
      }});
      rows += '</div>';
    }}
    portfolioResult.innerHTML = rows;
  }}

  var targetAmountInput = document.getElementById('target-amount-input');
  var targetDaysInput = document.getElementById('target-days-input');
  var growthChartEl = document.getElementById('growth-plan-chart');
  var growthSummaryEl = document.getElementById('growth-plan-summary');

  function buildGrowthSvg(start, target, days, color) {{
    var width = 320, height = 100, padX = 6, padY = 12;
    var dailyRate = Math.pow(target / start, 1 / days) - 1;
    var n = Math.min(Math.max(Math.round(days), 1), 60);
    var points = [];
    for (var i = 0; i <= n; i++) {{
      var d = (days / n) * i;
      points.push(start * Math.pow(1 + dailyRate, d));
    }}
    var lo = Math.min.apply(null, points);
    var hi = Math.max.apply(null, points);
    var span = (hi - lo) || 1;
    var step = (width - 2 * padX) / n;
    var coords = points.map(function (v, i) {{
      return [padX + i * step, height - padY - (v - lo) / span * (height - 2 * padY)];
    }});
    var pathD = 'M ' + coords.map(function (p) {{ return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }}).join(' L ');
    var areaD = pathD + ' L ' + coords[coords.length - 1][0].toFixed(1) + ' ' + (height - padY) +
      ' L ' + coords[0][0].toFixed(1) + ' ' + (height - padY) + ' Z';
    var last = coords[coords.length - 1];
    var svg = '<svg viewBox="0 0 ' + width + ' ' + height + '" width="100%" height="' + height +
      '" preserveAspectRatio="none" role="img" aria-label="想定成長曲線">' +
      '<path d="' + areaD + '" fill="' + color + '" opacity="0.12" stroke="none" />' +
      '<path d="' + pathD + '" fill="none" stroke="' + color + '" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />' +
      '<circle cx="' + last[0].toFixed(1) + '" cy="' + last[1].toFixed(1) + '" r="3.2" fill="' + color + '" stroke="var(--surface)" stroke-width="1.5" />' +
      '</svg>';
    return {{ svg: svg, dailyRate: dailyRate }};
  }}

  function bestAchievableDailyRate() {{
    if (!candidateData.length) return null;
    var rates = candidateData
      .filter(function (c) {{ return c.daysToProfit > 0; }})
      .map(function (c) {{ return Math.pow(1 + TAKE_PROFIT_PCT, 1 / c.daysToProfit) - 1; }});
    return rates.length ? Math.max.apply(null, rates) : null;
  }}

  function renderGrowthPlan() {{
    if (!growthChartEl || !growthSummaryEl) return;
    var start = parseFloat(budgetInput.value) || 0;
    var target = parseFloat(targetAmountInput.value) || 0;
    var days = parseFloat(targetDaysInput.value) || 0;
    var feasEl = document.getElementById('growth-plan-feasibility');

    if (start <= 0 || target <= start || days <= 0) {{
      growthChartEl.innerHTML = '';
      growthSummaryEl.textContent = '予算・目標金額・日数を正しく入力してください（目標は予算より大きい金額）。';
      if (feasEl) feasEl.innerHTML = '';
      return;
    }}

    var built = buildGrowthSvg(start, target, days, 'var(--accent)');
    growthChartEl.innerHTML = built.svg +
      '<div class="sparkline-caption"><span>今日 ' + yen(start) + '円</span><span>' + days + '営業日後 ' + yen(target) + '円</span></div>';

    var dailyPct = (built.dailyRate * 100).toFixed(2);
    var weeklyPct = ((Math.pow(1 + built.dailyRate, 5) - 1) * 100).toFixed(1);
    growthSummaryEl.textContent = '必要なペース: 1営業日あたり平均+' + dailyPct + '%（週あたり+' + weeklyPct + '%相当）の複利成長';

    if (feasEl) {{
      var bestRate = bestAchievableDailyRate();
      var html;
      if (bestRate === null) {{
        html = '<p>比較できる候補データがありません。</p>';
      }} else if (built.dailyRate <= bestRate * 0.5) {{
        html = '<p class="feasibility-tag feasibility-ok">現実的な範囲</p>' +
          '<p>現在見つかっている候補の最速ペース（1日あたり+' + (bestRate * 100).toFixed(2) + '%相当）の半分以下で済むペースです。' +
          '「短期×がっつり」型を中心に、上のポートフォリオ通り複数銘柄に分散すれば狙える可能性があります。</p>';
      }} else if (built.dailyRate <= bestRate) {{
        html = '<p class="feasibility-tag feasibility-hard">かなり強気</p>' +
          '<p>現在見つかっている最有力候補（1日あたり+' + (bestRate * 100).toFixed(2) + '%相当）とほぼ同じか、それ以上のペースが必要です。' +
          '1銘柄の保有だけでなく、利確ラインに届いたら即座に利益確定→次の「短期×がっつり」候補に乗り換える、を繰り返す' +
          '短期集中売買（数日単位のスイング〜デイトレード）が前提になります。</p>';
      }} else {{
        html = '<p class="feasibility-tag feasibility-unrealistic">非常に困難</p>' +
          '<p>現在見つかっている候補の最速ペース（1日あたり+' + (bestRate * 100).toFixed(2) + '%相当）を上回っており、' +
          '通常の「買って様子見」ではまず届きません。狙うなら、値動きの荒い銘柄でデイトレード（1日の中で売買を完結させる）を' +
          '繰り返して小さな利益を積み重ねる以外に現実的な道はほぼなく、それでも損失リスクは非常に高い点に注意してください。' +
          '目標を下げる、または期間を延ばすことも検討してください。</p>';
      }}
      feasEl.innerHTML = html;
    }}

    var achieveEl = document.getElementById('growth-plan-achieve');
    if (achieveEl) {{
      var stats = getActivePortfolioStats(start);
      if (!stats || stats.totalCost <= 0 || stats.weightedDays <= 0) {{
        achieveEl.innerHTML = '<p>比較できるポートフォリオがありません（品質ゲートを満たす候補がないか、まだ銘柄を選んでいません）。</p>';
      }} else {{
        var portfolioDailyRate = Math.pow((stats.totalCost + stats.totalProfit) / stats.totalCost, 1 / stats.weightedDays) - 1;
        var projected = stats.totalCost * Math.pow(1 + portfolioDailyRate, days);
        var sourceLabel = stats.source === 'selected'
          ? '選んだ' + stats.count + '銘柄'
          : 'おすすめポートフォリオ（' + stats.count + '銘柄、未選択のため自動計算分）';
        if (projected >= target) {{
          achieveEl.innerHTML = '<p class="feasibility-tag feasibility-ok">' + sourceLabel + 'なら達成できる可能性あり</p>' +
            '<p>このペース（1日あたり+' + (portfolioDailyRate * 100).toFixed(2) + '%相当）が続けば、' + days + '営業日後には約' +
            yen(projected) + '円が見込め、目標' + yen(target) + '円に届く計算です。</p>';
        }} else {{
          var shortfall = target - projected;
          var neededDays = portfolioDailyRate > 0 ? Math.log(target / stats.totalCost) / Math.log(1 + portfolioDailyRate) : null;
          var daysAdvice = (neededDays && neededDays > 0 && isFinite(neededDays))
            ? '(1) 期間を約' + Math.ceil(neededDays) + '営業日まで延ばす、'
            : '(1) このペースでは期間を延ばしても届かないため銘柄構成を見直す、';
          achieveEl.innerHTML = '<p class="feasibility-tag feasibility-hard">' + sourceLabel + 'のペースでは不足</p>' +
            '<p>このペースだと' + days + '営業日後は約' + yen(projected) + '円の見込みで、目標まであと' + yen(shortfall) + '円足りません。' +
            '対策: ' + daysAdvice + '(2) 予算を増やして購入株数を増やす、(3) 上の「短期×がっつり」型の銘柄比率を増やす、' +
            'のいずれかが考えられます。</p>';
        }}
      }}
    }}
  }}

  function recalcAll() {{
    var budget = parseFloat(budgetInput.value) || 0;
    recalcBudgetBoxes(budget);
    renderPortfolio();
    renderGrowthPlan();
  }}

  if (budgetInput) {{
    budgetInput.addEventListener('input', recalcAll);
    if (targetAmountInput) targetAmountInput.addEventListener('input', renderGrowthPlan);
    if (targetDaysInput) targetDaysInput.addEventListener('input', renderGrowthPlan);
    recalcAll();
  }}

  // --- タブ切り替え（全銘柄 / 選んだ銘柄でポートフォリオ） ---
  var viewAll = document.getElementById('view-all');
  var viewSelected = document.getElementById('view-selected');
  document.querySelectorAll('.view-tab-btn').forEach(function (btn) {{
    btn.addEventListener('click', function () {{
      document.querySelectorAll('.view-tab-btn').forEach(function (b) {{ b.classList.remove('is-active'); }});
      btn.classList.add('is-active');
      var view = btn.getAttribute('data-view');
      if (viewAll) viewAll.style.display = (view === 'all') ? '' : 'none';
      if (viewSelected) viewSelected.style.display = (view === 'selected') ? '' : 'none';
    }});
  }});

  // --- チェックボックスで選んだ銘柄の記録（このブラウザのlocalStorageにのみ保存） ---
  var PICKS_KEY = 'investmentPicks';

  function getPicks() {{
    try {{ return JSON.parse(localStorage.getItem(PICKS_KEY) || '{{}}'); }} catch (e) {{ return {{}}; }}
  }}
  function savePicks(picks) {{
    try {{ localStorage.setItem(PICKS_KEY, JSON.stringify(picks)); }} catch (e) {{ /* ignore */ }}
  }}
  function updatePickedCount() {{
    var el = document.getElementById('picked-count');
    if (!el) return;
    var n = Object.keys(getPicks()).length;
    el.textContent = n ? '（' + n + '）' : '';
  }}

  function currentPriceFor(ticker) {{
    var found = candidateData.filter(function (c) {{ return c.ticker === ticker; }})[0];
    return found ? found.price : null;
  }}

  function renderSelectedPortfolio() {{
    var container = document.getElementById('selected-portfolio-result');
    if (!container) return;
    var picks = getPicks();
    var tickers = Object.keys(picks);
    if (!tickers.length) {{
      container.innerHTML = '<p class="empty-msg">まだ銘柄が選ばれていません。「全銘柄」タブでチェックを入れてください。</p>';
      return;
    }}

    var html = '';
    tickers.forEach(function (ticker) {{
      var pick = picks[ticker];
      var checkPrice = pick.price;
      var current = currentPriceFor(ticker);
      var stopPrice = checkPrice * (1 + STOP_LOSS_PCT);
      var targetPrice = checkPrice * (1 + TAKE_PROFIT_PCT);
      var checkedAtStr = new Date(pick.checkedAt).toLocaleString('ja-JP');

      var currentBlock;
      if (current !== null) {{
        var changePct = ((current - checkPrice) / checkPrice) * 100;
        var cls = changePct >= 0 ? 'rise' : 'fall';
        currentBlock = '<span class="' + cls + '">現在値 ' + yen(current) + '円（チェック時から' + (changePct >= 0 ? '+' : '') + changePct.toFixed(2) + '%）</span>';
      }} else {{
        currentBlock = '<span class="portfolio-detail">現在値: 今回のスキャン対象外のため不明（このダッシュボードを更新すると再取得されます）</span>';
      }}

      html += '<div class="selected-card">' +
        '<div class="selected-card-top">' +
        '<span class="portfolio-name">' + pick.name + '<span class="ticker">' + ticker + '</span></span>' +
        '<button type="button" class="selected-remove-btn" data-remove="' + ticker + '">選択解除</button>' +
        '</div>' +
        '<div class="detail">チェック時点の価格 ' + yen(checkPrice) + '円（' + checkedAtStr + '）</div>' +
        '<div class="detail">' + currentBlock + '</div>' +
        '<div class="timing"><div class="timing-row"><span class="timing-label">売るタイミング</span>' +
        '<span>' + yen(stopPrice) + '円を下回ったら損切り、' + yen(targetPrice) + '円を上回ったら利確が目安' +
        '（チェック時点の価格基準）。</span></div></div>' +
        '</div>';
    }});
    container.innerHTML = html;

    container.querySelectorAll('.selected-remove-btn').forEach(function (btn) {{
      btn.addEventListener('click', function () {{
        var picks = getPicks();
        delete picks[btn.getAttribute('data-remove')];
        savePicks(picks);
        var cb = document.querySelector('.pick-checkbox[data-ticker="' + btn.getAttribute('data-remove') + '"]');
        if (cb) cb.checked = false;
        updatePickedCount();
        renderSelectedPortfolio();
        renderGrowthPlan();
      }});
    }});
  }}

  var existingPicks = getPicks();
  document.querySelectorAll('.pick-checkbox').forEach(function (cb) {{
    if (existingPicks[cb.dataset.ticker]) cb.checked = true;
    cb.addEventListener('change', function () {{
      var picks = getPicks();
      if (cb.checked) {{
        picks[cb.dataset.ticker] = {{
          name: cb.dataset.name,
          price: parseFloat(cb.dataset.price),
          checkedAt: new Date().toISOString(),
        }};
      }} else {{
        delete picks[cb.dataset.ticker];
      }}
      savePicks(picks);
      updatePickedCount();
      renderSelectedPortfolio();
      renderGrowthPlan();
    }});
  }});

  updatePickedCount();
  renderSelectedPortfolio();
}})();
</script>
"""


def load_market_scan_results(positions, benchmark_returns):
    """market_scan.py が見つけた候補・全銘柄基本情報を読み込む。
    候補は通常のウォッチリストと同じ形式で詳細分析し直し、
    全銘柄基本情報（all_stocks）は検索窓でそのまま使うため軽量なまま返す。"""
    if not MARKET_CANDIDATES_FILE.exists():
        return [], None, []

    try:
        data = json.loads(MARKET_CANDIDATES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[警告] market_candidates.json の読み込みに失敗: {e}")
        return [], None, []

    all_candidates = data.get("candidates", [])
    all_stocks = data.get("all_stocks", [])
    scan_meta = {
        "scanned_at": data.get("scanned_at"),
        "universe_size": data.get("universe_size"),
        "total_found": len(all_candidates),
    }
    watchlist_set = set(config.WATCHLIST)

    # 候補は勢いスコア順に並んでいる前提。多すぎると重いので上位だけ詳細分析する
    top_candidates = [c for c in all_candidates if c.get("ticker") not in watchlist_set][:config.MARKET_SCAN_TOP_N]

    market_finds = []
    for c in top_candidates:
        ticker = c.get("ticker")
        if not ticker:
            continue
        config.TICKER_NAMES.setdefault(ticker, c.get("name", ticker))
        info = analyze_ticker(ticker, positions, benchmark_returns)
        if info.get("status") == "error":
            continue
        market_finds.append(info)

    return market_finds, scan_meta, all_stocks


def generate_dashboard():
    # PUBLIC_BUILD=1 はGitHub Pages公開用のビルド（誰でも見られる）を意味する。
    # 保有銘柄の取得価格・含み損益は個人情報なので、公開ビルドでは常に「未保有」として扱う。
    if os.environ.get("PUBLIC_BUILD") == "1":
        positions = {}
    else:
        positions = load_positions()
    benchmark_returns = fetch_benchmark_returns()
    results = [analyze_ticker(ticker, positions, benchmark_returns) for ticker in config.WATCHLIST]
    market_finds, scan_meta, all_stocks = load_market_scan_results(positions, benchmark_returns)
    html = render_html(results, market_finds=market_finds, scan_meta=scan_meta, all_stocks=all_stocks)
    output_path = SCRIPT_DIR / "dashboard.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"ダッシュボード生成完了: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_dashboard()
