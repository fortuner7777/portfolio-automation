import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta

def run_pipeline():
    print("🚀 [시스템] 포트폴리오 자동 최적화 파이프라인 가동...")
    
    us_tickers = [
        'NVDA', 'MSFT', 'GOOGL', 'AVGO', 'TSLA', 'AMD', 'TSM', 'AMZN', 'META', 'IONQ',
        'TLN', 'COIN', 'CLS', 'BMNR', 'MU', 'ARM', 'UNH', 'VST', 'APP',
        'NET', 'CRWV', 'ORCL', 'ALAB', 'ANET', 'CRDO', 'INOD', 'RXRX', 'PGY'
    ]
    kr_tickers = ['005930.KS', '000660.KS', '035420.KS', '035720.KS']
    all_tickers = us_tickers + kr_tickers
    rename_map = {'005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '035420.KS': 'NAVER', '035720.KS': '카카오'}

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    
    # 데이터 정리: 모든 열을 확인하여 유효한 데이터만 남김
    if 'Adj Close' in raw_data.columns:
        price_data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        price_data = raw_data['Close']
    else:
        price_data = raw_data.xs('Adj Close', level=0, axis=1) if 'Adj Close' in raw_data.columns.levels[0] else raw_data.xs('Close', level=0, axis=1)

    # 1. 텅 빈 종목 제거, 2. 변동성이 0인 종목 제거 (ZeroDivision 방지)
    price_data = price_data.rename(columns=rename_map).dropna(axis=1, how='any')
    price_data = price_data.loc[:, price_data.std() > 0]

    daily_returns = price_data.pct_change().dropna()
    exp_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    
    assets = price_data.columns
    num_assets = len(assets)
    rf_rate = 0.035

    def get_perf(w):
        r = np.sum(exp_returns * w)
        v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        # 변동성이 0이면 매우 작은 값을 주어 에러 방지
        v = v if v > 1e-6 else 1e-6
        return r, v, (r - rf_rate) / v

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    res = minimize(lambda w: -get_perf(w)[2], np.array([1./num_assets]*num_assets), method='SLSQP', bounds=bounds, constraints=cons)
    ms_w = res.x
    ms_ret, ms_vol, ms_sharpe = get_perf(ms_w)

    df_w = pd.DataFrame({'Asset': assets, 'Weight': ms_w * 100})
    top_assets = df_w[df_w['Weight'] > 0.1].sort_values(by='Weight', ascending=False)

    report_date = datetime.now().strftime('%Y-%m-%d')
    msg = f"🌟 [{report_date}] 데일리 포트폴리오 최적화 리포트\n\n"
    msg += f"📊 [최대 샤프지수 포트폴리오 상태]\n"
    msg += f"• 기대수익률: {ms_ret*100:.1f}%\n"
    msg += f"• 포트폴리오 변동성: {ms_vol*100:.1f}%\n"
    msg += f"• 샤프 지수: {ms_sharpe:.2f}\n\n"
    msg += f"🔥 [추천 투자 비중 Top 자산]\n"
    for _, row in top_assets.iterrows():
        msg += f"• {row['Asset']}: {row['Weight']:.1f}%\n"

    sender_email = os.getenv("SENDER_EMAIL")
    app_password = os.getenv("EMAIL_APP_PASSWORD")
    receiver_email = os.getenv("RECEIVER_EMAIL")

    if sender_email and app_password and receiver_email:
        try:
            em = MIMEMultipart()
            em['From'] = sender_email
            em['To'] = receiver_email
            em['Subject'] = f"📈 {report_date} 포트폴리오 자동 분석 리포트"
            em.attach(MIMEText(msg, 'plain'))
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, app_password)
            server.send_message(em)
            server.quit()
            print("✅ [알림 발송 완료] 이메일이 성공적으로 전송되었습니다.")
        except Exception as e:
            print(f"❌ [알림 발송 실패] 에러: {e}")
    else:
        print("⚠️ [환경변수 미설정] 설정 확인 필요.")

if __name__ == "__main__":
    run_pipeline()
