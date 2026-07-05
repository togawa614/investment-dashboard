"""
投資部 - 売買シグナル検知スクリプト

やること:
1. config.py のウォッチリストにある個別株の株価データを取得する
2. 「ゴールデンクロス + 出来高急増」で買いシグナルを判定する
3. 保有中の銘柄（positions.json に記録）は、デッドクロス・損切り・利確で売りシグナルを判定する
4. シグナルが出たら keiyou233501@gmail.com 宛にメールで通知する
5. 実行結果は logs/YYYY-MM-DD.md に毎回記録する（発注は自動で行わない。判断と発注は必ず本人が行う）
"""

import json
import os
import smtplib
from datetime import datetime, date
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import yfinance as yf

import config

SCRIPT_DIR = Path(__file__).resolve().parent
POSITIONS_PATH = SCRIPT_DIR / config.POSITIONS_FILE
LOG_DIR = (SCRIPT_DIR / config.LOG_DIR).resolve()


def load_positions():
    if POSITIONS_PATH.exists():
        with open(POSITIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def fetch_history(ticker):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
    except Exception as e:
        print(f"[警告] {ticker} の取得に失敗: {e}")
        return None
    if data.empty:
        return None
    return data


def compute_signals(df):
    """短期MA・長期MA・出来高平均・クロス発生を計算して列に追加する"""
    df = df.copy()
    df["MA_SHORT"] = df["Close"].rolling(config.SHORT_MA_WINDOW).mean()
    df["MA_LONG"] = df["Close"].rolling(config.LONG_MA_WINDOW).mean()
    df["VOL_AVG"] = df["Volume"].rolling(config.VOLUME_LOOKBACK_WINDOW).mean()
    df["ABOVE"] = (df["MA_SHORT"] > df["MA_LONG"]).astype(int)
    df["CROSS"] = df["ABOVE"].diff()  # 1: ゴールデンクロス発生, -1: デッドクロス発生
    return df


def days_since_golden_cross(df):
    """直近のゴールデンクロスから何営業日経過しているか（見つからなければNone）"""
    cross_points = df.index[df["CROSS"] == 1]
    if len(cross_points) == 0:
        return None
    last_cross = cross_points[-1]
    return df.index.get_loc(df.index[-1]) - df.index.get_loc(last_cross)


def check_fundamentals(ticker):
    """売上成長率が大きく落ち込んでいないかだけを軽くチェックする。
    新興・ベンチャー株はデータが取得できないことが多いため、その場合は通過させる。"""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"[警告] {ticker} のファンダメンタル情報取得に失敗: {e}")
        return True

    revenue_growth = info.get("revenueGrowth")
    if revenue_growth is None:
        return True
    return revenue_growth >= config.FUNDAMENTAL_MIN_REVENUE_GROWTH


def detect_trend_tags(ticker):
    """直近ニュース見出しにテーマキーワードが含まれるかチェックする（参考情報・実験的機能）。
    シグナルの成立条件にはしない。取得失敗時は空リストを返す。"""
    try:
        news_items = yf.Ticker(ticker).news[: config.NEWS_LOOKBACK_COUNT]
    except Exception as e:
        print(f"[警告] {ticker} のニュース取得に失敗: {e}")
        return []

    tags = set()
    for item in news_items:
        if not isinstance(item, dict):
            continue
        title = item.get("title", "") or item.get("content", {}).get("title", "")
        for keyword in config.THEME_KEYWORDS:
            if keyword.lower() in title.lower():
                tags.add(keyword)
    return sorted(tags)


def check_buy_signal(ticker):
    df = fetch_history(ticker)
    min_len = config.LONG_MA_WINDOW + config.VOLUME_LOOKBACK_WINDOW
    if df is None or len(df) < min_len:
        return None, "データ不足"

    df = compute_signals(df)
    latest = df.iloc[-1]

    days_after_cross = days_since_golden_cross(df)
    if days_after_cross is None or days_after_cross > config.GOLDEN_CROSS_RECENT_DAYS:
        return None, None

    if pd.isna(latest["VOL_AVG"]) or latest["VOL_AVG"] == 0:
        return None, None

    volume_ratio = latest["Volume"] / latest["VOL_AVG"]
    if volume_ratio < config.VOLUME_MULTIPLIER:
        return None, None

    if not check_fundamentals(ticker):
        return None, None

    return {
        "ticker": ticker,
        "price": round(float(latest["Close"]), 2),
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "volume_ratio": round(float(volume_ratio), 2),
        "days_after_cross": int(days_after_cross),
        "trend_tags": detect_trend_tags(ticker),
    }, None


def check_sell_signal(ticker, position):
    df = fetch_history(ticker)
    if df is None:
        return None, "データ取得失敗"

    df = compute_signals(df)
    latest = df.iloc[-1]
    price = float(latest["Close"])
    entry_price = float(position["entry_price"])
    change_pct = (price - entry_price) / entry_price

    reasons = []

    lookback = config.GOLDEN_CROSS_RECENT_DAYS + 1
    recent_cross = df["CROSS"].iloc[-lookback:]
    if (recent_cross == -1).any():
        reasons.append("デッドクロス発生")

    if change_pct <= config.STOP_LOSS_PCT:
        reasons.append(f"損切りライン到達（{change_pct:.1%}）")

    if change_pct >= config.TAKE_PROFIT_PCT:
        reasons.append(f"利確ライン到達（{change_pct:.1%}）")

    if not reasons:
        return None, None

    return {
        "ticker": ticker,
        "price": round(price, 2),
        "entry_price": entry_price,
        "change_pct": round(change_pct * 100, 2),
        "reasons": reasons,
        "date": df.index[-1].strftime("%Y-%m-%d"),
    }, None


def send_email(subject, body):
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_password:
        raise RuntimeError(
            "環境変数 GMAIL_APP_PASSWORD が設定されていません。"
            "GmailのアプリパスワードをセットしてからCLIから実行してください（README.md参照）。"
        )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = config.MAIL_FROM
    msg["To"] = config.MAIL_TO

    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.MAIL_FROM, app_password)
        server.send_message(msg)


def append_log(lines):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"{today_str}.md"
    now_str = datetime.now().strftime("%H:%M")

    is_new = not log_path.exists()
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                f'---\ndate: "{today_str}"\ntype: signal-log\n---\n\n'
                f"# {today_str} シグナルチェックログ\n\n"
            )
        f.write(f"## {now_str}\n")
        for line in lines:
            f.write(f"- {line}\n")
        f.write("\n")


def main():
    positions = load_positions()
    buy_signals = []
    sell_signals = []
    log_lines = []

    for ticker in config.WATCHLIST:
        if ticker in positions:
            signal, error = check_sell_signal(ticker, positions[ticker])
            if error:
                log_lines.append(f"{ticker}: エラー - {error}")
            elif signal:
                sell_signals.append(signal)
                log_lines.append(f"{ticker}: 売りシグナル ({', '.join(signal['reasons'])})")
            else:
                log_lines.append(f"{ticker}: 保有中・シグナルなし")
        else:
            signal, error = check_buy_signal(ticker)
            if error:
                log_lines.append(f"{ticker}: エラー - {error}")
            elif signal:
                buy_signals.append(signal)
                log_lines.append(f"{ticker}: 買いシグナル発生")
            else:
                log_lines.append(f"{ticker}: シグナルなし")

    append_log(log_lines)

    if not buy_signals and not sell_signals:
        print("シグナルなし。ログに記録しました。")
        return

    body_lines = []
    if buy_signals:
        body_lines.append("■ 買いシグナル")
        for s in buy_signals:
            trend_str = f" / トレンドタグ: {', '.join(s['trend_tags'])}" if s["trend_tags"] else ""
            body_lines.append(
                f"- {s['ticker']}: 終値 {s['price']}円 / 出来高倍率 {s['volume_ratio']}倍 "
                f"/ ゴールデンクロスから{s['days_after_cross']}営業日 ({s['date']}){trend_str}"
            )
        body_lines.append("")
    if sell_signals:
        body_lines.append("■ 売りシグナル")
        for s in sell_signals:
            body_lines.append(
                f"- {s['ticker']}: 終値 {s['price']}円 / 購入価格 {s['entry_price']}円 "
                f"/ 変化率 {s['change_pct']}% / 理由: {', '.join(s['reasons'])} ({s['date']})"
            )
        body_lines.append("")

    body_lines.append("※これは投資助言ではありません。最終判断はご自身で行い、SBI証券で手動発注してください。")

    subject = f"【投資部】シグナル通知 {date.today().strftime('%Y-%m-%d')}"
    send_email(subject, "\n".join(body_lines))
    print("シグナルを検知し、メールを送信しました。")


if __name__ == "__main__":
    main()
