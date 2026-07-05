"""
投資部 - 東証の全上場銘柄から「買いシグナル候補」を探すスクリプト

やること:
1. JPX（日本取引所グループ）が公開している上場銘柄一覧（data_j.xls）をダウンロードする
2. プライム・スタンダード・グロース市場の内国株式（約3,700銘柄）を対象に、
   ゴールデンクロス＋出来高急増＋「通常100株を予算2万円以内で買える」という条件で一次スクリーニングする
   （全銘柄でファンダメンタル判定や理論株価まで計算すると時間がかかりすぎるため、
   まずは技術的な条件だけで候補を絞り込む。100株買うのに2万円を超える銘柄はそもそも対象外にする）
3. 見つかった候補の銘柄コードを market_candidates.json に保存する
4. dashboard.py がこのファイルを読み込み、候補銘柄の詳細（現状・タイミング・理論株価など）を
   通常のウォッチリストと同じ形式で追加表示する

このスクリプトは重い処理（全銘柄ダウンロード）なので、毎回のダッシュボード更新ではなく
時々（週1回など）実行する運用を想定。
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

import config

SCRIPT_DIR = Path(__file__).resolve().parent
LISTING_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
LISTING_CACHE = SCRIPT_DIR / "data_j.xls"
CANDIDATES_FILE = SCRIPT_DIR / "market_candidates.json"

TARGET_MARKETS = ["プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）"]
CHUNK_SIZE = 400
HISTORY_PERIOD = "3mo"


def fetch_listing():
    """上場銘柄一覧をダウンロード（既に今日分があればそれを使う）"""
    if LISTING_CACHE.exists():
        age_days = (time.time() - LISTING_CACHE.stat().st_mtime) / 86400
        if age_days < 7:
            return pd.read_excel(LISTING_CACHE)

    resp = requests.get(LISTING_URL, timeout=60)
    resp.raise_for_status()
    LISTING_CACHE.write_bytes(resp.content)
    return pd.read_excel(LISTING_CACHE)


def get_universe():
    df = fetch_listing()
    mask = df["市場・商品区分"].isin(TARGET_MARKETS)
    subset = df.loc[mask, ["コード", "銘柄名"]].copy()
    subset["ticker"] = subset["コード"].astype(str) + ".T"
    return dict(zip(subset["ticker"], subset["銘柄名"]))


def screen_chunk(data, tickers):
    """ダウンロード済みの株価データから、ゴールデンクロス＋出来高急増の銘柄だけ抽出する。
    後で上位だけに絞れるよう、出来高倍率とトレンドの強さも一緒に返す。"""
    hits = []
    min_len = config.LONG_MA_WINDOW + config.VOLUME_LOOKBACK_WINDOW

    for t in tickers:
        try:
            sub = data[t][["Close", "Volume"]].dropna()
        except (KeyError, TypeError):
            continue
        if len(sub) < min_len:
            continue

        ma_short = sub["Close"].rolling(config.SHORT_MA_WINDOW).mean()
        ma_long = sub["Close"].rolling(config.LONG_MA_WINDOW).mean()
        vol_avg = sub["Volume"].rolling(config.VOLUME_LOOKBACK_WINDOW).mean()

        above = (ma_short > ma_long).astype(int)
        cross = above.diff()
        cross_points = sub.index[cross == 1]
        if len(cross_points) == 0:
            continue

        last_cross = cross_points[-1]
        days_after = sub.index.get_loc(sub.index[-1]) - sub.index.get_loc(last_cross)
        if days_after > config.GOLDEN_CROSS_RECENT_DAYS:
            continue

        latest_vol = sub["Volume"].iloc[-1]
        avg_vol = vol_avg.iloc[-1]
        if pd.isna(avg_vol) or avg_vol == 0:
            continue
        volume_ratio = latest_vol / avg_vol
        if volume_ratio < config.VOLUME_MULTIPLIER:
            continue

        # 通常の100株単位で予算内（2万円）で買えない銘柄はそもそも候補にしない
        price = float(sub["Close"].iloc[-1])
        if price * 100 > config.BUDGET_JPY:
            continue

        trend_strength = float(ma_short.iloc[-1] / ma_long.iloc[-1] - 1) if ma_long.iloc[-1] else 0.0
        momentum_score = trend_strength + (volume_ratio - 1) * 0.3

        hits.append({
            "ticker": t,
            "price": round(price, 1),
            "volume_ratio": round(float(volume_ratio), 2),
            "days_after_cross": int(days_after),
            "momentum_score": round(momentum_score, 4),
        })

    return hits


def scan_market():
    universe = get_universe()
    tickers = list(universe.keys())
    print(f"スキャン対象: {len(tickers)}銘柄")

    all_hits = []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        print(f"  {i}〜{i + len(chunk)}件目をダウンロード中...")
        try:
            data = yf.download(
                tickers=chunk, period=HISTORY_PERIOD, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"  [警告] チャンク取得に失敗: {e}")
            continue

        if len(chunk) == 1:
            # 1銘柄だけだとMultiIndexにならないので、その形に合わせて包み直す
            data = pd.concat({chunk[0]: data}, axis=1)

        hits = screen_chunk(data, chunk)
        all_hits.extend(hits)

    all_hits.sort(key=lambda h: h["momentum_score"], reverse=True)
    for h in all_hits:
        h["name"] = universe.get(h["ticker"], h["ticker"])

    result = {
        "scanned_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "universe_size": len(tickers),
        "candidates": all_hits,
    }
    CANDIDATES_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完了: {len(all_hits)}件の候補が見つかりました（勢いスコア順）→ {CANDIDATES_FILE}")
    return result


if __name__ == "__main__":
    scan_market()
