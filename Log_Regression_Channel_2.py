import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- 1. 網頁配置與自定義 CSS ---
st.set_page_config(page_title="David 長線股價對數回歸通道", layout="wide")

# --- 2. 側邊欄：參數設定 ---
st.sidebar.header("查詢設定")

# 股票代號
stock_id = st.sidebar.text_input("股票代號(如2330.TW或AAPL)", "2330")

# 日期選擇
start_date = st.sidebar.date_input("起始日期", datetime(2015, 8, 1))
end_date = st.sidebar.date_input("結束日期", datetime.now())

# 圖表主題選擇
theme_choice = st.sidebar.radio("圖表主題(對應網頁背景)", ["亮色(白色背景)", "深色(深色背景)"])

# --- 3. 強制背景色與字體加深邏輯 (CSS) ---
if theme_choice == "深色(深色背景)":
    chart_template = "plotly_dark"
    font_color = "white"
    bg_color = "#0E1117"
    st.markdown("""
        <style>
        /* 強制側邊欄、主背景、文字顏色為深色 */
        [data-testid="stSidebar"], .stApp, header { background-color: #0E1117 !important; color: white !important; }
        .stMarkdown, p, h1, h2, h3, span { color: white !important; }
        /* 調整輸入框文字顏色 */
        input { color: white !important; background-color: #262730 !important; }
        </style>
        """, unsafe_allow_html=True)
else:
    chart_template = "plotly_white"
    font_color = "#000000" # 加深為純黑
    bg_color = "#FFFFFF"
    st.markdown("""
        <style>
        /* 1. 強制背景與文字顏色 */
        [data-testid="stSidebar"], .stApp, header { 
            background-color: #FFFFFF !important; 
            color: black !important; 
        }
        .stMarkdown, p, h1, h2, h3, span { color: black !important; }
        
        /* 2. 徹底消除輸入框右側的陰影與淡淡格線 */
        div[data-baseweb="input"], 
        div[data-baseweb="input"] > div,
        div[data-baseweb="input"] input {
            background-color: white !important;
            border-color: #dcdcdc !important; /* 設定一個淺灰色的統一邊框 */
            box-shadow: none !important;      /* 移除所有陰影 */
        }
        
        /* 針對日期選取器內部的特殊容器進行修正 */
        div[role="combobox"] {
            background-color: white !important;
            border: none !important;
        }

        /* 3. 強制按鈕內部的所有文字元素變白 */
        div.stButton > button {
            background-color: #000000 !important;
            border: 1px solid #000000 !important;
            font-weight: bold !important;
        }
        div.stButton > button * {
            color: #FFFFFF !important;
        }
        
        div.stButton > button:hover {
            background-color: #333333 !important;
        }

        /* 4. 側邊欄與輸入框整體調整 */
        [data-testid="stSidebar"] { border-right: 1px solid #f0f2f6; }
        input { 
            color: black !important; 
            background-color: white !important; 
        }
        </style>
        """, unsafe_allow_html=True)

# 定義開始計算按鈕
calculate_btn = st.sidebar.button("開始計算")

# --- 4. 主要標題 ---
st.write(f"## 📈 David 長線股價對數回歸通道")

if not calculate_btn:
    st.info("💡 請點開左上角選單 [ >> ] 在左側面板設定參數後，按「開始計算」即可產出圖表")
else:
    # A. 下載資料（自動依序嘗試原始代號、.TW、.TWO）
    raw_id = stock_id.strip()
    candidates = [raw_id, f"{raw_id}.TW", f"{raw_id}.TWO"] if '.' not in raw_id else [raw_id]

    data = pd.DataFrame()
    search_id = raw_id
    for candidate in candidates:
        fetched = yf.download(candidate, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if not fetched.empty:
            data = fetched
            search_id = candidate
            break

    if not data.empty:
        # 取得公司名稱
        # ticker_info = yf.Ticker(search_id) #先拿掉避免觸發YFRateLimitError
        # long_name = ticker_info.info.get('longName', search_id) #先拿掉避免觸發YFRateLimitError
        # st.write(f"### {search_id} - {long_name}") #先拿掉避免觸發YFRateLimitError
        st.write(f"### {search_id}")

        # B. 資料處理與多層索引處理
        # 先展平 MultiIndex 欄位 (新版 yfinance 對單一股票也可能回傳 MultiIndex)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 移除重複欄位 (MultiIndex 展平後可能出現)
        data = data.loc[:, ~data.columns.duplicated()]

        df = data.reset_index()

        # 相容不同版本 yfinance 的日期欄位名稱
        # 新版 yfinance (0.2.x+) 的索引名稱可能為 'Datetime' 而非 'Date'
        if 'Date' not in df.columns:
            date_col_found = None
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_col_found = col
                    break
            if date_col_found:
                df = df.rename(columns={date_col_found: 'Date'})
            else:
                # 最後備援：將第一欄重新命名為 'Date'
                df = df.rename(columns={df.columns[0]: 'Date'})

        # 確保 Date 欄位為 datetime 型別
        df['Date'] = pd.to_datetime(df['Date'])

        # 安全建立收盤價欄位
        # 新版 yfinance 的 Close 欄位名稱可能帶有大小寫變化
        close_col = next((c for c in df.columns if c.lower() == 'close'), None)
        if close_col is None:
            st.error("找不到收盤價欄位，請檢查代號或日期。")
            st.stop()
        df['Close_1D'] = df[close_col]

        # [核心] 1. 對數化股價：對收盤價取對數
        df['Log_Close'] = np.log(df['Close_1D'])

        # 格式化日期字串 (用於 X 軸消除缺口)
        df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')

        # [核心] 2. 線性回歸計算 (針對對數化股價)
        X = np.array(df.index).reshape(-1, 1)
        Y = df['Log_Close'].values.reshape(-1, 1)
        
        # 排除 NaN 進行回歸訓練
        mask = ~np.isnan(Y).flatten()
        model = LinearRegression()
        model.fit(X[mask], Y[mask])
        
        # 預測對數回歸值 (中心線)
        df['Log_Reg'] = model.predict(X)

        # [核心] 3. 計算離差與標準差 (SD)
        # 離差 = 對數實際股價 - 對數回歸值
        df['Deviation'] = df['Log_Close'] - df['Log_Reg']
        sd_val = df['Deviation'].std()

        # [核心] 4. 計算五線譜軌道 (對數空間)
        df['Log_P2SD'] = df['Log_Reg'] + (2 * sd_val)
        df['Log_P1SD'] = df['Log_Reg'] + sd_val
        df['Log_M1SD'] = df['Log_Reg'] - sd_val
        df['Log_M2SD'] = df['Log_Reg'] - (2 * sd_val)

        # C. 繪圖：使用 Plotly
        fig = go.Figure()

        # 軌道線繪製 (對數化數據)
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Log_Close'], name='對數化股價', line=dict(color='#17BECF', width=2)))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Log_Reg'], name='回歸中線', line=dict(color='orange', dash='dash')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Log_P2SD'], name='+2SD 極端樂觀', line=dict(color='red', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Log_P1SD'], name='+1SD 樂觀', line=dict(color='pink', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Log_M1SD'], name='-1SD 悲觀', line=dict(color='lightgreen', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Log_M2SD'], name='-2SD 極端悲觀', line=dict(color='green', width=1, dash='dot')))

        # D. 圖表佈局設定 (強制加深字體、消除格線)
        fig.update_layout(
            height=650,
            template=chart_template,
            hovermode="x unified",
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            # 1. 全域字體設定：強制設為 font_color (純黑) 並放大
            font=dict(color=font_color, size=14), 
            
            # 2. 強制指定圖例文字顏色與大小
            legend_font_color=font_color,
            legend_font_size=14,
            
            xaxis=dict(
                type='category', 
                color=font_color, 
                tickfont=dict(color=font_color, size=12),
                nticks=10,
                showgrid=False, 
                zeroline=False
            ),
            
            yaxis=dict(
                color=font_color, 
                tickfont=dict(color=font_color, size=12),
                title=dict(text="對數化股價 Log(Price)", font=dict(color=font_color, size=14)),
                showgrid=False, 
                zeroline=False
            ),
            
            # 3. 圖例詳細設定
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5,
                # 這裡再次強制指定顏色，並增加邊框讓它更明顯
                font=dict(color=font_color, size=14),
                bordercolor=font_color, # 增加一個淡淡的邊框有助於視覺識別
                borderwidth=0,
                itemsizing='constant' # 讓圖示大小固定，不會變細
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # E. 數據摘要區
        st.header("📊 最後交易日數據摘要")
        last_row = df.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("原始收盤價", f"{last_row['Close_1D']:.2f}")
        col2.metric("對數化股價", f"{last_row['Log_Close']:.4f}")
        col3.metric("對數回歸值", f"{last_row['Log_Reg']:.4f}")
        col4.metric("對數離差 SD", f"{sd_val:.4f}")
        
    else:
        st.error(f"找不到股票資料（已嘗試：{', '.join(candidates)}），請檢查代號或日期。")
