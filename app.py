# 🚀 뉴스팩토리 - AI 기반 기사 분석 플랫폼 (konlpy 버전 + 최적화)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
import time
import hashlib
from pathlib import Path
import requests
import pickle
import json
from collections import Counter, defaultdict
import subprocess
import sys

# Java 환경변수 설정 (konlpy 사용 전)
if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# konlpy import는 환경변수 설정 후에
from konlpy.tag import Okt
okt = Okt()

# 머신러닝 라이브러리 import
def safe_import():
    """개선된 안전한 라이브러리 import"""
    global TfidfVectorizer, cosine_similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        st.success("✅ 머신러닝 라이브러리 로드 성공")
    except ImportError:
        st.warning("⚠️ scikit-learn이 설치되지 않았습니다.")
        TfidfVectorizer = None
        cosine_similarity = None

safe_import()

# 구글 시트 유틸리티 모듈
try:
    from google_sheets_utils_deploy import (
        load_data_from_google_sheets_cached,
        preprocess_dataframe,
        get_connection_status
    )
    SHEETS_UTILS_AVAILABLE = True
except ImportError:
    SHEETS_UTILS_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="🚀 뉴스팩토리",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 전역 설정
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
APP_PASSWORD = st.secrets["APP_PASSWORD"]

# 캐시 설정
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
API_CACHE_DIR = CACHE_DIR / "api_calls"
API_CACHE_DIR.mkdir(exist_ok=True)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 세션 상태 초기화
# ================================
def initialize_session_state():
    """세션 상태 초기화"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'main_df' not in st.session_state:
        st.session_state.main_df = None
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'login_status' not in st.session_state:
        st.session_state.login_status = False

# ================================
# 캐싱 시스템
# ================================
def save_to_cache(key, data):
    """캐시에 데이터 저장"""
    try:
        cache_file = API_CACHE_DIR / f"{key}.pkl"
        cache_data = {
            'data': data,
            'timestamp': time.time(),
            'date': datetime.now().strftime("%Y-%m-%d")
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        return True
    except Exception as e:
        st.error(f"캐시 저장 실패: {str(e)}")
        return False

def load_from_cache(key, max_age_hours=24):
    """캐시에서 데이터 로드"""
    try:
        cache_file = API_CACHE_DIR / f"{key}.pkl"
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 캐시 유효성 검사
        cache_age = time.time() - cache_data['timestamp']
        if cache_age > max_age_hours * 3600:
            return None
        
        return cache_data['data']
    except:
        return None

# ================================
# 데이터 로딩 함수 (세션 상태 관리 강화)
# ================================
def load_data_with_session_management(force_refresh=False):
    """세션 상태를 활용한 데이터 로딩"""
    # 이미 로드된 데이터가 있고 강제 새로고침이 아닌 경우
    if not force_refresh and st.session_state.data_loaded and st.session_state.main_df is not None:
        st.info("💾 저장된 데이터를 사용합니다.")
        return st.session_state.main_df
    
    # 새로운 데이터 로드
    with st.spinner("📊 구글 시트에서 데이터 로딩 중..."):
        if SHEETS_UTILS_AVAILABLE:
            df = load_data_from_google_sheets_cached(force_refresh=force_refresh)
            if not df.empty:
                df = preprocess_dataframe(df)
                # 세션 상태에 저장
                st.session_state.main_df = df
                st.session_state.data_loaded = True
                st.session_state.last_update = datetime.now()
                st.success(f"✅ 데이터 로딩 완료: {len(df)}개 기사")
                return df
            else:
                st.error("❌ 구글 시트에서 데이터를 가져올 수 없습니다.")
                return pd.DataFrame()
        else:
            st.error("❌ 구글 시트 연결 모듈이 없습니다.")
            return pd.DataFrame()

# ================================
# API 호출
# ================================
def call_perplexity_api_cached(prompt, model="sonar-pro", max_age_hours=24):
    """캐싱된 Perplexity API 호출"""
    try:
        # 캐시 키 생성
        cache_key = hashlib.md5(f"{prompt}_{model}".encode()).hexdigest()
        
        # 캐시에서 먼저 확인
        cached_result = load_from_cache(cache_key, max_age_hours)
        if cached_result:
            st.info("🔄 캐시에서 AI 분석 결과를 불러왔습니다.")
            return cached_result
        
        # API 호출
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 뉴스 전문 분석가입니다. 주어진 기사 데이터를 분석하여 핵심 트렌드와 중요한 키워드를 한국어로 간결하게 요약해주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            api_result = result["choices"][0]["message"]["content"]
            
            # 결과를 캐시에 저장
            save_to_cache(cache_key, api_result)
            st.success("🆕 새로운 AI 분석을 완료했습니다.")
            return api_result
        else:
            st.error(f"API 오류: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"API 호출 실패: {str(e)}")
        return None

# ================================
# 핵심 분석 기능들
# ================================
def generate_today_review(df, industry="전체"):
    """📊 오늘의 리뷰 - 당일 주요 기사 종합 분석"""
    if df.empty:
        return "분석할 데이터가 없습니다."
    
    # 업계 필터링
    filtered_df = df if industry == "전체" else df[df["업계"] == industry]
    if filtered_df.empty:
        return "해당 업계의 기사가 없습니다."
    
    # 오늘 날짜 기준 최근 기사들 선별
    today = datetime.now()
    current_month = today.month
    
    # 이번 달 기사들만 필터링
    if '월' in filtered_df.columns:
        current_month_articles = filtered_df[filtered_df['월'] == f"{current_month}월"]
    else:
        current_month_articles = filtered_df
    
    # 상위 가중치 기사들 선별
    top_articles = current_month_articles.nlargest(15, '전체가중치')
    
    # 기사 텍스트 정리
    review_texts = []
    for _, article in top_articles.iterrows():
        title = article.get('제목', '')
        content = article.get('주요내용', '')
        media = article.get('매체', '')
        if pd.notna(title):
            review_texts.append(f"[{media}] {title}\n요약: {str(content)[:200]}")
    
    if not review_texts:
        return "분석할 기사가 없습니다."
    
    prompt = f"""
    당신은 경험 많은 뉴스 에디터입니다. 
    오늘의 주요 뉴스들을 종합하여 '오늘의 뉴스 리뷰'를 작성해주세요.
    
    [분석 대상: {industry} 업계]
    
    다음 형식으로 작성해주세요:
    
    ## 📊 오늘의 핵심 이슈
    - 주요 이슈 3가지를 간결하게 정리
    
    ## 📈 주목할 트렌드
    - 오늘 기사들에서 발견된 트렌드 분석
    
    ## 🔍 심층 분석 포인트
    - 추가 주목해야 할 지점들
    
    ## 📋 내일 주목할 키워드
    - 향후 관련 기사에서 주목할 키워드들
    
    기사 목록:
    {chr(10).join(review_texts)}
    """
    
    return call_perplexity_api_cached(prompt, max_age_hours=6)

def generate_weekly_monthly_review(df, period_type="주차", target_period=None):
    """🗓️ 월·주차 리뷰 - 기간별 트렌드 종합 분석"""
    if df.empty:
        return "분석할 데이터가 없습니다."
    
    if period_type == "월":
        if target_period and "월" in df.columns:
            period_data = df[df["월"] == target_period]
            period_name = target_period
        else:
            # 현재 월 데이터
            current_month = f"{datetime.now().month}월"
            period_data = df[df["월"] == current_month] if "월" in df.columns else df
            period_name = current_month
    else:  # 주차
        if target_period and "주차" in df.columns:
            period_data = df[df["주차"] == target_period]
            period_name = target_period
        else:
            # 최근 주차 데이터
            if "주차" in df.columns:
                latest_week = df["주차"].dropna().iloc[-1] if not df["주차"].dropna().empty else "이번주"
                period_data = df[df["주차"] == latest_week]
                period_name = latest_week
            else:
                period_data = df
                period_name = "이번 기간"
    
    if period_data.empty:
        return f"{period_name} 데이터가 없습니다."
    
    # 주요 기사들 선별
    top_articles = period_data.nlargest(20, '전체가중치')
    
    # 매체별 분포 분석
    media_distribution = period_data['매체'].value_counts().head(5)
    
    # 업계별 분포 분석
    industry_distribution = period_data['업계'].value_counts().head(5)
    
    # 텍스트 정리
    article_texts = []
    for _, article in top_articles.iterrows():
        title = article.get('제목', '')
        media = article.get('매체', '')
        industry = article.get('업계', '')
        if pd.notna(title):
            article_texts.append(f"[{media}|{industry}] {title}")
    
    prompt = f"""
    당신은 뉴스 트렌드 분석 전문가입니다.
    {period_name}의 뉴스 동향을 종합 분석하여 상세한 리뷰를 작성해주세요.
    
    ## 📊 {period_name} 리뷰 리포트
    
    다음 형식으로 작성해주세요:
    
    ### 🎯 주요 이슈 Top 3
    - 이 기간의 가장 중요한 이슈 3가지
    
    ### 📈 트렌드 분석
    - 주목할 만한 트렌드와 변화
    
    ### 📰 매체별 특징
    - 주요 매체들의 보도 경향 분석
    
    ### 🏢 업계별 동향
    - 업계별 주요 뉴스 특징
    
    ### 🔮 향후 전망
    - 다음 기간 주목할 포인트
    
    [분석 데이터]
    - 총 기사 수: {len(period_data)}개
    - 주요 매체: {', '.join(media_distribution.index[:3])}
    - 주요 업계: {', '.join(industry_distribution.index[:3])}
    
    주요 기사 목록:
    {chr(10).join(article_texts)}
    """
    
    return call_perplexity_api_cached(prompt, max_age_hours=12)

def generate_custom_analysis(df, keywords, industry="전체", analysis_type="종합"):
    """🎯 맞춤 분석 - 키워드 기반 심층 분석"""
    if df.empty or not keywords:
        return "분석할 데이터나 키워드가 없습니다."
    
    # 업계 필터링
    filtered_df = df if industry == "전체" else df[df["업계"] == industry]
    if filtered_df.empty:
        return "해당 업계의 기사가 없습니다."
    
    # 키워드 리스트 처리
    if isinstance(keywords, str):
        keyword_list = [k.strip() for k in keywords.split(',')]
    else:
        keyword_list = keywords
    
    # 키워드가 포함된 기사들 필터링
    keyword_articles = []
    for _, article in filtered_df.iterrows():
        title = str(article.get('제목', '')).lower()
        content = str(article.get('주요내용', '')).lower()
        combined_text = f"{title} {content}"
        
        # 키워드 매칭 확인
        matches = sum(1 for keyword in keyword_list if keyword.lower() in combined_text)
        if matches > 0:
            article_dict = article.to_dict()
            article_dict['keyword_matches'] = matches
            keyword_articles.append(article_dict)
    
    if not keyword_articles:
        return f"'{', '.join(keyword_list)}' 키워드와 관련된 기사를 찾을 수 없습니다."
    
    # 키워드 매칭 수와 가중치 기준으로 정렬
    keyword_articles.sort(key=lambda x: (x['keyword_matches'], x.get('전체가중치', 0)), reverse=True)
    
    # 상위 기사들 선별
    top_keyword_articles = keyword_articles[:15]
    
    # 분석용 텍스트 준비
    analysis_texts = []
    for article in top_keyword_articles:
        title = article.get('제목', '')
        media = article.get('매체', '')
        matches = article.get('keyword_matches', 0)
        if title:
            analysis_texts.append(f"[{media}] {title} (매칭: {matches}개)")
    
    # 분석 타입별 프롬프트
    if analysis_type == "트렌드":
        analysis_focus = "시간의 흐름에 따른 키워드 관련 트렌드 변화와 향후 전망"
    elif analysis_type == "매체비교":
        analysis_focus = "매체별 보도 관점과 논조의 차이점 분석"
    elif analysis_type == "영향분석":
        analysis_focus = "해당 키워드가 업계와 사회에 미치는 영향 분석"
    else:  # 종합
        analysis_focus = "키워드 관련 종합적인 현황과 의미 분석"
    
    prompt = f"""
    당신은 뉴스 분석 전문가입니다.
    '{', '.join(keyword_list)}' 키워드에 대한 맞춤형 심층 분석을 수행해주세요.
    
    [분석 조건]
    - 대상 업계: {industry}
    - 분석 타입: {analysis_type}
    - 분석 초점: {analysis_focus}
    - 관련 기사 수: {len(keyword_articles)}개
    
    다음 형식으로 분석해주세요:
    
    ## 🎯 키워드 분석: {', '.join(keyword_list)}
    
    ### 📊 현황 개요
    - 키워드 관련 주요 이슈 정리
    
    ### 🔍 심층 분석
    - {analysis_focus}
    
    ### 📈 주요 발견사항
    - 분석을 통해 발견한 핵심 인사이트
    
    ### 💡 시사점
    - 해당 키워드의 의미와 향후 관심 포인트
    
    ### 🔮 후속 관찰 포인트
    - 계속 지켜봐야 할 관련 이슈들
    
    분석 대상 기사:
    {chr(10).join(analysis_texts)}
    """
    
    return call_perplexity_api_cached(prompt, max_age_hours=8)

# ================================
# 로그인 함수
# ================================
def login():
    """로그인 처리"""
    st.title("🔐 뉴스팩토리 로그인")
    
    password = st.text_input("비밀번호를 입력하세요", type="password")
    
    if st.button("로그인", key="login_button"):
        if password == APP_PASSWORD:
            st.session_state.login_status = True
            st.success("✅ 로그인 성공!")
            st.rerun()
        else:
            st.error("❌ 비밀번호가 틀렸습니다.")

# ================================
# 메인 애플리케이션
# ================================
def main():
    """메인 애플리케이션"""
    # 세션 상태 초기화
    initialize_session_state()
    
    # 로그인 확인
    if not st.session_state.login_status:
        login()
        return
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 뉴스팩토리</h1>
        <p>AI 기반 기사 분석 플랫폼</p>
        <p>✨ konlpy 기반 한국어 처리 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 제어판")
        
        # 데이터 새로고침 버튼
        if st.button("🔄 데이터 새로고침", key="refresh_data_button"):
            st.session_state.data_loaded = False
            st.session_state.main_df = None
            st.rerun()
        
        # 데이터 로드 상태 표시
        if st.session_state.data_loaded and st.session_state.last_update:
            st.success(f"📊 데이터 로드됨")
            st.info(f"🕒 마지막 업데이트: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 구글 시트 연결 상태
        if SHEETS_UTILS_AVAILABLE:
            connection_status = get_connection_status()
            if connection_status['connected']:
                st.success("✅ 구글 시트 연결됨")
            else:
                st.error(f"❌ 구글 시트 연결 실패: {connection_status['message']}")
    
    # 데이터 로드
    df = load_data_with_session_management()
    
    if df.empty:
        st.error("❌ 데이터를 불러올 수 없습니다.")
        return
    
    # 메인 기능 탭
    tab1, tab2, tab3 = st.tabs([
        "📊 오늘의 리뷰", "🗓️ 월·주차 리뷰", "🎯 맞춤 분석"
    ])
    
    with tab1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 오늘의 리뷰</h3>
            <p>당일 주요 기사들을 종합 분석하여 핵심 이슈와 트렌드를 정리합니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 업계 선택
        industries = ['전체'] + sorted(df['업계'].unique().tolist())
        selected_industry = st.selectbox("업계 선택", industries, key="today_review_industry")
        
        if st.button("📊 오늘의 리뷰 생성", use_container_width=True, key="generate_today_review"):
            with st.spinner("🤖 AI가 오늘의 뉴스를 종합 분석하는 중..."):
                today_review = generate_today_review(df, selected_industry)
            
            st.markdown("### 📊 오늘의 뉴스 리뷰")
            st.markdown(today_review)
    
    with tab2:
        st.markdown("""
        <div class="feature-card">
            <h3>🗓️ 월·주차 리뷰</h3>
            <p>특정 기간의 뉴스 동향을 종합 분석하여 트렌드 리포트를 제공합니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 기간 타입 선택
        col1, col2 = st.columns(2)
        with col1:
            period_type = st.selectbox("분석 기간", ["주차", "월"], key="period_type_select")
        with col2:
            if period_type == "월" and "월" in df.columns:
                available_months = sorted(df["월"].dropna().unique().tolist())
                target_period = st.selectbox("월 선택", ["현재 월"] + available_months, key="target_month_select")
                if target_period == "현재 월":
                    target_period = None
            elif period_type == "주차" and "주차" in df.columns:
                available_weeks = sorted(df["주차"].dropna().unique().tolist())
                target_period = st.selectbox("주차 선택", ["최근 주차"] + available_weeks, key="target_week_select")
                if target_period == "최근 주차":
                    target_period = None
            else:
                target_period = None
                st.info("해당 기간 데이터가 없습니다.")
        
        if st.button("🗓️ 기간별 리뷰 생성", use_container_width=True, key="generate_period_review"):
            with st.spinner(f"🤖 AI가 {period_type} 리뷰를 생성하는 중..."):
                period_review = generate_weekly_monthly_review(df, period_type, target_period)
            
            st.markdown(f"### 🗓️ {period_type} 리뷰 리포트")
            st.markdown(period_review)
    
    with tab3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 맞춤 분석</h3>
            <p>특정 키워드나 주제에 대한 심층 분석을 제공합니다.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 분석 설정
        col1, col2 = st.columns(2)
        with col1:
            # 업계 선택
            industries = ['전체'] + sorted(df['업계'].unique().tolist())
            selected_industry = st.selectbox("업계 선택", industries, key="custom_analysis_industry")
            
            # 분석 타입 선택
            analysis_type = st.selectbox(
                "분석 유형", 
                ["종합", "트렌드", "매체비교", "영향분석"], 
                key="analysis_type_select"
            )
        
        with col2:
            # 키워드 입력
            keywords_input = st.text_input(
                "분석 키워드", 
                placeholder="예: 인공지능, 전기차, 바이오 (쉼표로 구분)",
                key="custom_keywords_input"
            )
            
            # 키워드 개수 표시
            if keywords_input:
                keyword_count = len([k.strip() for k in keywords_input.split(',') if k.strip()])
                st.info(f"입력된 키워드: {keyword_count}개")
        
        # 분석 실행
        if keywords_input:
            if st.button("🎯 맞춤 분석 실행", use_container_width=True, key="execute_custom_analysis"):
                keywords_list = [k.strip() for k in keywords_input.split(',') if k.strip()]
                
                with st.spinner(f"🤖 AI가 '{', '.join(keywords_list)}' 키워드를 심층 분석하는 중..."):
                    custom_analysis_result = generate_custom_analysis(
                        df, keywords_list, selected_industry, analysis_type
                    )
                
                st.markdown("### 🎯 맞춤 분석 결과")
                st.markdown(custom_analysis_result)
        else:
            if st.button("🎯 맞춤 분석 실행", use_container_width=True, key="execute_custom_analysis_empty"):
                st.warning("⚠️ 분석할 키워드를 입력해주세요.")

if __name__ == "__main__":
    main()
