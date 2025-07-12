# 🚀 뉴스팩토리 - AI 기반 기사 분석 플랫폼 (konlpy 버전 + 백업파일 기능 복원)
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
import calendar

# Java 환경변수 설정 (konlpy 사용 전)
if not os.environ.get("JAVA_HOME"):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# konlpy import는 환경변수 설정 후에
from konlpy.tag import Okt
okt = Okt()

# 자동 패키지 설치 함수
def install_package(package):
    """패키지 자동 설치"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

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
        test_connection,
        get_google_sheets_client,
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
DATA_CACHE_DIR = CACHE_DIR / "data"
DATA_CACHE_DIR.mkdir(exist_ok=True)
ANALYSIS_CACHE_DIR = CACHE_DIR / "analysis"
ANALYSIS_CACHE_DIR.mkdir(exist_ok=True)
API_CACHE_DIR = CACHE_DIR / "api_calls"
API_CACHE_DIR.mkdir(exist_ok=True)

# 어두운 테마 CSS 스타일
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
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
def get_cache_key(data_type, industry, date_str=None):
    """캐시 키 생성"""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    key_string = f"{data_type}_{industry}_{date_str}"
    return hashlib.md5(key_string.encode()).hexdigest()

def save_to_cache(cache_type, key, data):
    """캐시에 데이터 저장"""
    try:
        if cache_type == "data":
            cache_file = DATA_CACHE_DIR / f"{key}.pkl"
        elif cache_type == "analysis":
            cache_file = ANALYSIS_CACHE_DIR / f"{key}.pkl"
        else:
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

def load_from_cache(cache_type, key, max_age_hours=24):
    """캐시에서 데이터 로드"""
    try:
        if cache_type == "data":
            cache_file = DATA_CACHE_DIR / f"{key}.pkl"
        elif cache_type == "analysis":
            cache_file = ANALYSIS_CACHE_DIR / f"{key}.pkl"
        else:
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
# 데이터 로딩 함수 (세션 상태 관리)
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
        cached_result = load_from_cache("api", cache_key, max_age_hours)
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
            save_to_cache("api", cache_key, api_result)
            st.success("🆕 새로운 AI 분석을 완료했습니다.")
            return api_result
        else:
            st.error(f"API 오류: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"API 호출 실패: {str(e)}")
        return None

# ================================
# 🔧 NEW UTILS (새로운 기능 함수들)
# ================================
def compile_articles_text(articles, max_each=3):
    """유사 기사 리스트 → Prompt용 텍스트 문자열"""
    texts = []
    for art in articles[:max_each]:
        title = art.get("제목", "")
        body = art.get("주요내용", "")
        if pd.notna(title):
            texts.append(f"제목: {title}\n내용: {str(body)[:300]}")
    return "\n".join(texts)

def generate_monthly_ai_plans(insight_dict, df, industry="전체"):
    """
    주차별 핵심문장 + 유사 기사들을 묶어 Perplexity API로 'AI 추천 기획 아이템' 생성
    """
    if not insight_dict or df.empty:
        return "데이터가 부족해 기획 아이템을 생성할 수 없습니다."
    
    plan_source_blocks = []
    for week, sentence in insight_dict.get("weekly_insights", {}).items():
        if not sentence:
            continue
        sims = find_similar_articles_to_insight(df, sentence, 7)
        sim_text = compile_articles_text([s["article"] for s in sims])
        plan_source_blocks.append(
            f"[{week}] 핵심문장: {sentence}\n관련 기사:\n{sim_text}"
        )
    
    prompt = f"""
    당신은 한국 주요 일간지의 {industry if industry!='전체' else ''} 담당 부장입니다.
    아래 주차별 핵심 트렌드와 관련 기사들을 분석해 다음 달 취재 방향을 제시할 'AI 추천 기획 아이템' 5건을 작성하세요.
    
    [요청 형식]
    제시 문장(기사 제목 후보) - 한 줄 이유
    
    [분석 대상 데이터]
    {os.linesep.join(plan_source_blocks[:3])}
    """
    
    return call_perplexity_api_cached(prompt, max_age_hours=6)

def generate_custom_topic_brief(df, industry, topic, weight_threshold=0.45):
    """
    업계+주제 관련 기사 추출 후 Perplexity API로 고급 브리핑 생성
    """
    if df.empty or not topic:
        return None
    
    # 업계 필터
    pool = df if industry == "전체" else df[df["업계"] == industry]
    if pool.empty:
        return None
    
    # 유사도 계산
    pool = pool.copy()
    pool["combined_text"] = pool["제목"].fillna("") + " " + pool["주요내용"].fillna("")
    
    if not TfidfVectorizer or not cosine_similarity:
        return "TF-IDF 분석 라이브러리가 필요합니다."
    
    vec = TfidfVectorizer(max_features=1500)
    try:
        mat = vec.fit_transform(pool["combined_text"].tolist() + [topic])
        sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
        
        pool["sim"] = sims
        cand = pool[(pool["sim"] > 0.1) & (pool["전체가중치"] >= weight_threshold)]
        
        if cand.empty:
            return None
        
        cand = cand.sort_values(["sim", "전체가중치"], ascending=False).head(20)
        
        joined = [
            f"제목: {r['제목']}\n내용: {str(r['주요내용'])[:300]}"
            for _, r in cand.iterrows()
        ]
        
        prompt = f"""
        당신은 한국 언론사의 {industry if industry!='전체' else ''} 담당 편집국장입니다.
        주제: {topic}
        
        아래 기사들을 바탕으로
        1) 현재 핵심 트렌드 3가지
        2) 각 트렌드별 매체 보도 특징
        3) 앞으로 발굴해야 할 심층 기획 기사 3건(제목+간단 이유)
        을 기자들에게 지시하는 구어체 메모 형식으로 작성하세요.
        
        기사 목록:
        {os.linesep.join(joined[:10])}
        """
        
        return call_perplexity_api_cached(prompt, max_age_hours=6)
        
    except:
        return "분석 중 오류가 발생했습니다."

# ================================
# 새로운 백업 기능들 추가
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
# 새로운 AI 기능 함수들
# ================================
def generate_today_planning_items(df, industry="전체"):
    """
    상위 가중치 기사들을 바탕으로 Perplexity API를 호출, '오늘의 기획 아이템' 3개를 제안한다.
    """
    if df.empty:
        return "분석할 데이터가 없습니다."
    
    if industry != "전체":
        df = df[df["업계"] == industry]
    
    # 가중치 상위 기사 20개까지 대상
    top_articles = (
        df.nlargest(20, "전체가중치") if "전체가중치" in df.columns else df.head(20)
    )
    
    texts = []
    for _, art in top_articles.iterrows():
        title = art.get("제목", "")
        body = art.get("주요내용", "")
        if pd.notna(title) and pd.notna(body):
            texts.append(f"제목: {title}\n내용: {body}")
        elif pd.notna(title):
            texts.append(f"제목: {title}")
    
    if not texts:
        return "분석할 기사 내용이 없습니다."
    
    prompt = f"""
    다음은 오늘의 주요 뉴스 기사들입니다. 기자들이 추가 취재하기 좋은 '기획 아이템' 3건을 제안해주세요.
    
    기사 목록:
    {os.linesep.join(texts[:15])}
    
    [답변 형식]
    1. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    
    2. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    
    3. 기획 아이템 제목:
    - 취재 포인트:
    - 예상 소스:
    """
    
    return (
        call_perplexity_api_cached(prompt, max_age_hours=6)
        or "기획 아이템 생성에 실패했습니다."
    )

def generate_monthly_weekly_insights(df, target_month=None):
    """
    월별·주차별 핵심 문장 생성(PPLX API) → dict 반환
    {
        'monthly_insight': str,
        'weekly_insights': {주차: str, …}
    }
    """
    if df.empty:
        return {}
    
    if target_month and "월" in df.columns:
        df = df[df["월"] == target_month]
    
    if df.empty:
        return {}
    
    # 월 전체 핵심 문장
    month_txt = " ".join(
        (
            f"{r.get('제목','')} {r.get('주요내용','')}"
            for _, r in df.head(50).iterrows()
        )
    )
    
    mon_prompt = f"""
    다음은 {target_month or '이번 달'} 주요 뉴스입니다.
    핵심 이슈를 한 문장으로 요약해 주세요.
    
    뉴스 내용:
    {month_txt}
    """
    
    monthly_insight = call_perplexity_api_cached(mon_prompt, max_age_hours=12)
    
    # 주차별
    weekly_insights = {}
    if "주차" in df.columns:
        for week in sorted(df["주차"].dropna().unique()):
            w_df = df[df["주차"] == week]
            week_txt = " ".join(
                (
                    f"{r.get('제목','')} {r.get('주요내용','')}"
                    for _, r in w_df.head(20).iterrows()
                )
            )
            
            w_prompt = f"""
            다음은 {week} 주요 뉴스입니다.
            핵심 이슈를 한 문장으로 요약해 주세요.
            
            뉴스 내용:
            {week_txt}
            """
            
            weekly_insights[week] = call_perplexity_api_cached(
                w_prompt, max_age_hours=24
            )
    
    return {"monthly_insight": monthly_insight, "weekly_insights": weekly_insights}

def find_similar_articles_to_insight(df, insight, top_n=5):
    """
    핵심 문장(insight) 과 유사도가 높은 기사 n개 반환
    """
    if df.empty or not insight:
        return []
    
    df = df.copy()
    df["combined_text"] = df.apply(
        lambda r: f"{r.get('제목','')} {r.get('주요내용','')}", axis=1
    )
    
    df = df[df["combined_text"].str.strip() != ""]
    
    if df.empty or not (TfidfVectorizer and cosine_similarity):
        return []
    
    try:
        vec = TfidfVectorizer(max_features=1000)
        mat = vec.fit_transform(df["combined_text"].tolist() + [insight])
        sims = cosine_similarity(mat[-1:], mat[:-1]).flatten()
        
        idxs = sims.argsort()[::-1][:top_n]
        return [
            {"article": df.iloc[i], "similarity": sims[i]}
            for i in idxs
            if sims[i] > 0.1
        ]
    except:
        return []

# ================================
# konlpy 기반 키워드 추출 (soynlp 대체)
# ================================
def extract_keywords_okt(text_list, top_n=15):
    """konlpy Okt를 사용한 키워드 추출"""
    try:
        if not text_list:
            return []
        
        # 텍스트 전처리
        if isinstance(text_list, str):
            text_list = [text_list]
        
        # 텍스트 정제
        cleaned_texts = []
        for text in text_list:
            if pd.notna(text):
                # 기본 정제
                text = str(text).strip()
                text = re.sub(r'[^\w\s가-힣]', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                if len(text) > 10:  # 너무 짧은 텍스트 제외
                    cleaned_texts.append(text)
        
        if not cleaned_texts:
            return []
        
        # 전체 텍스트 결합
        corpus = ' '.join(cleaned_texts)
        
        # 명사 추출
        nouns = okt.nouns(corpus)
        
        # 명사 빈도 계산
        noun_freq = Counter(nouns)
        
        # 불용어 제거
        stopwords = {
            '기자', '사진', '제공', '관련', '업계', '시장', '분야', '회사', '기업',
            '발표', '공개', '설명', '보도', '뉴스', '기사', '오늘', '어제', '내일',
            '최근', '현재', '당시', '이번', '다음', '지난', '국내', '해외', '전국',
            '것', '수', '등', '및', '또', '더', '이', '그', '을', '를', '의', '가',
            '말했다', '밝혔다', '전했다', '했다', '있다', '없다', '한다', '된다'
        }
        
        # 최종 키워드 선별
        final_keywords = []
        for word, freq in noun_freq.most_common():
            if (word not in stopwords and 
                not word.isdigit() and 
                len(word) >= 2 and 
                len(word) <= 10 and 
                re.search(r'[가-힣]', word)):
                final_keywords.append((word, freq))
        
        return final_keywords[:top_n]
    
    except Exception as e:
        st.error(f"키워드 추출 실패: {str(e)}")
        return extract_keywords_simple(text_list, top_n)

def extract_keywords_simple(text_list, top_n=10):
    """백업용 간단한 키워드 추출"""
    try:
        if isinstance(text_list, str):
            text_list = [text_list]
        
        all_text = ' '.join([str(text) for text in text_list if pd.notna(text)])
        
        # 간단한 토큰화
        words = re.findall(r'[가-힣]{2,6}', all_text)
        
        # 불용어 제거
        stopwords = {
            '기자', '사진', '제공', '관련', '업계', '시장', '분야', '회사', '기업',
            '발표', '공개', '설명', '보도', '뉴스', '기사', '오늘', '어제', '내일'
        }
        
        keywords = [word for word in words if word not in stopwords]
        word_freq = Counter(keywords)
        
        return word_freq.most_common(top_n)
    except:
        return []

# 기존 extract_keywords_mecab 함수 대체
def extract_keywords_mecab(text, top_n=10):
    """키워드 추출 - konlpy Okt 사용"""
    if isinstance(text, str):
        return extract_keywords_okt([text], top_n)
    elif isinstance(text, list):
        return extract_keywords_okt(text, top_n)
    else:
        return extract_keywords_okt([str(text)], top_n)

# ================================
# 업계별 키워드 추출 기능
# ================================
def extract_industry_keywords(df, industry):
    """업계별 특화 키워드 추출"""
    if industry != '전체':
        industry_articles = df[df['업계'] == industry]
    else:
        industry_articles = df
    
    titles = industry_articles['제목'].dropna().tolist()
    
    # konlpy Okt로 업계 특화 키워드 추출
    keywords = extract_keywords_okt(titles, 20)
    
    # 업계별 가중치 조정
    industry_weights = {
        '자동차': ['전기차', '자율주행', '배터리', '모빌리티', '전동화'],
        'IT/전자': ['AI', '반도체', '클라우드', '디지털', '인공지능'],
        '금융': ['핀테크', '가상화폐', '블록체인', '투자', '디지털뱅킹'],
        '바이오': ['백신', '치료제', '임상', '신약', '의료'],
        '화학/에너지': ['친환경', '탄소중립', '재생에너지', '수소'],
        '항공/운송': ['물류', '배송', '모빌리티', '운송'],
        '건설/부동산': ['개발', '분양', '재건축', '리모델링'],
        '유통/소비재': ['이커머스', '온라인', '배송', '소비'],
        '엔터테인먼트/미디어': ['콘텐츠', '플랫폼', '스트리밍', 'OTT'],
        '게임': ['메타버스', '게임', '플랫폼', '모바일'],
        '식품/음료': ['건강', '프리미엄', '친환경', '유기농']
    }
    
    if industry in industry_weights:
        boost_keywords = industry_weights[industry]
        enhanced_keywords = []
        for word, score in keywords:
            if any(boost in word for boost in boost_keywords):
                enhanced_keywords.append((word, score * 1.5))
            else:
                enhanced_keywords.append((word, score))
        return sorted(enhanced_keywords, key=lambda x: x[1], reverse=True)
    
    return keywords

# ================================
# 네이버 링크 하이퍼링크 처리 함수 (핵심 수정 부분)
# ================================
def get_naver_link_column(df):
    """구글 시트에서 네이버 링크 컬럼명을 찾는 함수"""
    possible_columns = ['네이버링크', '네이버 링크', '네이버URL', '네이버 URL', 'naver_link', 'naver_url']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # 유사한 컬럼명 찾기
    for col in df.columns:
        if '네이버' in str(col) and ('링크' in str(col) or 'URL' in str(col)):
            return col
    
    return None

def get_media_link_column(df):
    """구글 시트에서 매체링크 컬럼명을 찾는 함수"""
    possible_columns = ['매체링크', '매체 링크', '매체URL', '매체 URL', 'media_link', 'media_url']
    
    for col in possible_columns:
        if col in df.columns:
            return col
    
    # 유사한 컬럼명 찾기
    for col in df.columns:
        if '매체' in str(col) and ('링크' in str(col) or 'URL' in str(col)):
            return col
    
    return None

def create_article_hyperlink(title, naver_url=None, media_url=None):
    """기사 제목에 하이퍼링크를 생성하는 함수"""
    if pd.isna(title):
        title = "제목 없음"
    else:
        title = str(title).strip()
    
    # 1순위: 네이버링크 사용
    if pd.notna(naver_url) and str(naver_url).strip() and str(naver_url).strip().lower() not in ['nan', '', 'none']:
        naver_url = str(naver_url).strip()
        if not naver_url.startswith(('http://', 'https://')):
            naver_url = 'https://' + naver_url
        return f'📰 [{title}]({naver_url})'
    
    # 2순위: 매체링크 사용
    if pd.notna(media_url) and str(media_url).strip() and str(media_url).strip().lower() not in ['nan', '', 'none']:
        media_url = str(media_url).strip()
        if not media_url.startswith(('http://', 'https://')):
            media_url = 'https://' + media_url
        return f'📰 [{title}]({media_url})'
    
    return f"📄 {title}"

def display_article_with_link(article, df=None):
    """기사를 하이퍼링크와 함께 표시하는 함수"""
    title = article.get('제목', '제목 없음')
    
    # 네이버 링크 컬럼명 자동 감지
    if df is not None:
        naver_col = get_naver_link_column(df)
        naver_url = article.get(naver_col) if naver_col else None
        
        # 매체 링크 컬럼명 자동 감지
        media_col = get_media_link_column(df)
        media_url = article.get(media_col) if media_col else None
    else:
        naver_url = article.get('네이버링크') or article.get('네이버 URL')
        media_url = article.get('매체링크') or article.get('매체 URL')
    
    # 하이퍼링크가 적용된 제목 생성
    linked_title = create_article_hyperlink(title, naver_url, media_url)
    
    return linked_title

# ================================
# PyArrow 오류 해결용 안전한 테이블 표시 함수
# ================================
def safe_display_dataframe(df, title="데이터"):
    """PyArrow 오류를 피한 안전한 데이터프레임 표시"""
    try:
        # 먼저 st.dataframe 시도
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        # PyArrow 오류 시 대안 표시
        st.warning(f"⚠️ 표 표시 오류 (PyArrow): {str(e)}")
        st.markdown(f"### 📊 {title}")
        
        # HTML 테이블로 대체
        html_table = "<table style='width:100%'><tr>"
        for col in df.columns:
            html_table += f"<th>{col}</th>"
        html_table += "</tr>"
        
        for _, row in df.head(10).iterrows():
            html_table += "<tr>"
            for col in df.columns:
                html_table += f"<td>{row[col]}</td>"
            html_table += "</tr>"
        html_table += "</table>"
        
        st.markdown(html_table, unsafe_allow_html=True)

# ================================
# 히스토그램 표시 함수 (수정된 부분)
# ================================
def display_histogram(data, title="분포"):
    """히스토그램을 안전하게 표시하는 함수"""
    try:
        # 방법 1: matplotlib 사용
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('값')
        ax.set_ylabel('빈도')
        
        st.pyplot(fig)
        plt.close()
        
    except ImportError:
        # matplotlib가 없는 경우 bar_chart 사용
        try:
            # 히스토그램 데이터를 bar_chart용으로 변환
            hist_data = pd.Series(data).value_counts().sort_index()
            st.bar_chart(hist_data)
        except:
            # 마지막 방법: 단순 bar_chart
            st.bar_chart(pd.Series(data))

# ================================
# 로그인 함수
# ================================
def login():
    """로그인 처리"""
    st.title("🔐 뉴스팩토리 로그인")
    
    password = st.text_input("비밀번호를 입력하세요", type="password")
    
    if st.button("로그인"):
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
        <p>✨ konlpy 기반 한국어 처리 시스템 + 백업파일 기능 복원</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 제어판")
        
        # 데이터 새로고침 버튼
        if st.button("🔄 데이터 새로고침"):
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
        
        # 캐시 정리
        if st.button("🗑️ 캐시 정리"):
            try:
                for cache_dir in [DATA_CACHE_DIR, ANALYSIS_CACHE_DIR, API_CACHE_DIR]:
                    for file in cache_dir.glob("*.pkl"):
                        file.unlink()
                st.success("✅ 캐시 정리 완료")
            except Exception as e:
                st.error(f"❌ 캐시 정리 실패: {str(e)}")
    
    # 데이터 로드
    df = load_data_with_session_management()
    
    if df.empty:
        st.error("❌ 데이터를 불러올 수 없습니다.")
        return
    
    # 메인 탭 (백업 기능 추가)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 데이터 분석", "🔍 키워드 분석", "🤖 AI 인사이트", "📈 트렌드 분석", 
        "🎯 기획 아이템", "📊 오늘의 리뷰", "🗓️ 월·주차 리뷰", "🎯 맞춤 분석"
    ])
    
    with tab1:
        st.subheader("📊 데이터 분석")
        
        # 업계 선택
        industries = ['전체'] + sorted(df['업계'].unique().tolist())
        selected_industry = st.selectbox("업계 선택", industries)
        
        # 필터링된 데이터
        filtered_df = df if selected_industry == '전체' else df[df['업계'] == selected_industry]
        
        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("전체 기사 수", len(filtered_df))
        with col2:
            st.metric("평균 가중치", f"{filtered_df['전체가중치'].mean():.3f}")
        with col3:
            st.metric("최고 가중치", f"{filtered_df['전체가중치'].max():.3f}")
        with col4:
            st.metric("매체 수", filtered_df['매체'].nunique())
        
        # 상위 기사 표시
        st.subheader("⭐ 상위 가중치 기사")
        top_articles = filtered_df.nlargest(10, '전체가중치')[['제목', '매체', '전체가중치']]
        
        # 링크와 함께 표시
        for idx, article in top_articles.iterrows():
            article_data = filtered_df.loc[idx]
            linked_title = display_article_with_link(article_data, df)
            st.markdown(f"**{linked_title}**")
            st.write(f"매체: {article['매체']} | 가중치: {article['전체가중치']:.3f}")
            st.write("---")
        
        # 매체별 분포
        st.subheader("📰 매체별 분포")
        media_dist = filtered_df['매체'].value_counts().head(10)
        st.bar_chart(media_dist)
        
        # 업계별 분포
        if selected_industry == '전체':
            st.subheader("🏢 업계별 분포")
            industry_dist = df['업계'].value_counts()
            st.bar_chart(industry_dist)
    
    with tab2:
        st.subheader("🔍 키워드 분석")
        
        # 업계 선택
        industries = ['전체'] + sorted(df['업계'].unique().tolist())
        selected_industry = st.selectbox("업계 선택", industries, key="keyword_industry")
        
        # 키워드 추출
        with st.spinner("🔍 키워드 추출 중..."):
            keywords = extract_industry_keywords(df, selected_industry)
        
        if keywords:
            # 키워드 데이터프레임 생성
            keyword_df = pd.DataFrame(keywords, columns=['키워드', '빈도'])
            
            st.subheader(f"🏷️ {selected_industry} 주요 키워드")
            
            # 키워드 차트
            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(keyword_df.set_index('키워드')['빈도'])
            with col2:
                st.dataframe(keyword_df)
        else:
            st.warning("⚠️ 키워드를 추출할 수 없습니다.")
    
    with tab3:
        st.subheader("🤖 AI 인사이트")
        
        # 업계 선택
        industries = ['전체'] + sorted(df['업계'].unique().tolist())
        selected_industry = st.selectbox("업계 선택", industries, key="ai_industry")
        
        # 오늘의 기획 아이템
        st.subheader("📋 오늘의 기획 아이템")
        if st.button("🤖 AI 기획 아이템 생성"):
            with st.spinner("🤖 AI 분석 중..."):
                planning_items = generate_today_planning_items(df, selected_industry)
            st.markdown(planning_items)
        
        # 월별/주차별 인사이트
        st.subheader("📅 월별/주차별 인사이트")
        if st.button("🤖 인사이트 생성"):
            with st.spinner("🤖 AI 분석 중..."):
                insights = generate_monthly_weekly_insights(df)
            
            if insights.get('monthly_insight'):
                st.markdown("### 📊 월별 핵심 인사이트")
                st.markdown(insights['monthly_insight'])
            
            if insights.get('weekly_insights'):
                st.markdown("### 📊 주차별 핵심 인사이트")
                for week, insight in insights['weekly_insights'].items():
                    st.markdown(f"**{week}**: {insight}")
        
        # 맞춤형 주제 브리핑
        st.subheader("🎯 맞춤형 주제 브리핑")
        topic = st.text_input("분석할 주제를 입력하세요", placeholder="예: 인공지능, 전기차, 바이오")
        
        if st.button("🤖 주제 브리핑 생성") and topic:
            with st.spinner("🤖 AI 분석 중..."):
                topic_brief = generate_custom_topic_brief(df, selected_industry, topic)
            if topic_brief:
                st.markdown(topic_brief)
    
    with tab4:
        st.subheader("📈 트렌드 분석")
        
        # 월별 트렌드
        if '월' in df.columns:
            st.subheader("📅 월별 기사 수")
            monthly_trend = df['월'].value_counts().sort_index()
            st.line_chart(monthly_trend)
        
        # 주차별 트렌드
        if '주차' in df.columns:
            st.subheader("📅 주차별 기사 수")
            weekly_trend = df['주차'].value_counts().sort_index()
            st.line_chart(weekly_trend)
        
        # 가중치 분포 (수정된 부분)
        st.subheader("⚖️ 가중치 분포")
        display_histogram(df['전체가중치'], "가중치 분포")
    
    with tab5:
        st.subheader("🎯 기획 아이템")
        
        # 업계 선택
        industries = ['전체'] + sorted(df['업계'].unique().tolist())
        selected_industry = st.selectbox("업계 선택", industries, key="planning_industry")
        
        # 기획 아이템 생성
        if st.button("🎯 월간 기획 아이템 생성"):
            with st.spinner("🤖 AI 기획 아이템 생성 중..."):
                # 월별/주차별 인사이트 먼저 생성
                insights = generate_monthly_weekly_insights(df)
                
                if insights:
                    # 기획 아이템 생성
                    planning_items = generate_monthly_ai_plans(insights, df, selected_industry)
                    st.markdown("### 📋 AI 추천 기획 아이템")
                    st.markdown(planning_items)
                else:
                    st.warning("⚠️ 인사이트를 생성할 수 없습니다.")
        
        # 유사 기사 검색
        st.subheader("🔍 유사 기사 검색")
        search_query = st.text_input("검색어를 입력하세요", placeholder="예: 인공지능 규제")
        
        if st.button("🔍 유사 기사 검색") and search_query:
            similar_articles = find_similar_articles_to_insight(df, search_query, 10)
            
            if similar_articles:
                st.markdown("### 🔍 검색 결과")
                for i, result in enumerate(similar_articles, 1):
                    article = result['article']
                    similarity = result['similarity']
                    
                    linked_title = display_article_with_link(article, df)
                    st.markdown(f"**{i}. {linked_title}**")
                    st.write(f"유사도: {similarity:.3f} | 매체: {article.get('매체', 'N/A')}")
                    if pd.notna(article.get('주요내용')):
                        st.write(f"내용: {str(article['주요내용'])[:200]}...")
                    st.write("---")
            else:
                st.warning("⚠️ 관련 기사를 찾을 수 없습니다.")
    
    # 백업 기능들 추가
    with tab6:
        st.subheader("📊 오늘의 리뷰")
        
        # 업계 선택
        industries = ['전체'] + sorted(df['업계'].unique().tolist())
        selected_industry = st.selectbox("업계 선택", industries, key="today_review_industry")
        
        st.markdown("""
        **📊 오늘의 리뷰**는 당일 주요 기사들을 종합 분석하여 핵심 이슈와 트렌드를 정리합니다.
        """)
        
        if st.button("📊 오늘의 리뷰 생성", use_container_width=True):
            with st.spinner("🤖 AI가 오늘의 뉴스를 종합 분석하는 중..."):
                today_review = generate_today_review(df, selected_industry)
            
            st.markdown("### 📊 오늘의 뉴스 리뷰")
            st.markdown(today_review)
    
    with tab7:
        st.subheader("🗓️ 월·주차 리뷰")
        
        st.markdown("""
        **🗓️ 월·주차 리뷰**는 특정 기간의 뉴스 동향을 종합 분석하여 트렌드 리포트를 제공합니다.
        """)
        
        # 기간 타입 선택
        col1, col2 = st.columns(2)
        with col1:
            period_type = st.selectbox("분석 기간", ["주차", "월"], key="period_type")
        with col2:
            if period_type == "월" and "월" in df.columns:
                available_months = sorted(df["월"].dropna().unique().tolist())
                target_period = st.selectbox("월 선택", ["현재 월"] + available_months, key="target_month")
                if target_period == "현재 월":
                    target_period = None
            elif period_type == "주차" and "주차" in df.columns:
                available_weeks = sorted(df["주차"].dropna().unique().tolist())
                target_period = st.selectbox("주차 선택", ["최근 주차"] + available_weeks, key="target_week")
                if target_period == "최근 주차":
                    target_period = None
            else:
                target_period = None
                st.info("해당 기간 데이터가 없습니다.")
        
        if st.button("🗓️ 기간별 리뷰 생성", use_container_width=True):
            with st.spinner(f"🤖 AI가 {period_type} 리뷰를 생성하는 중..."):
                period_review = generate_weekly_monthly_review(df, period_type, target_period)
            
            st.markdown(f"### 🗓️ {period_type} 리뷰 리포트")
            st.markdown(period_review)
    
    with tab8:
        st.subheader("🎯 맞춤 분석")
        
        st.markdown("""
        **🎯 맞춤 분석**은 특정 키워드나 주제에 대한 심층 분석을 제공합니다.
        """)
        
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
                key="analysis_type"
            )
        
        with col2:
            # 키워드 입력
            keywords_input = st.text_input(
                "분석 키워드", 
                placeholder="예: 인공지능, 전기차, 바이오 (쉼표로 구분)",
                key="custom_keywords"
            )
            
            # 키워드 개수 표시
            if keywords_input:
                keyword_count = len([k.strip() for k in keywords_input.split(',') if k.strip()])
                st.info(f"입력된 키워드: {keyword_count}개")
        
        # 분석 실행
        if st.button("🎯 맞춤 분석 실행", use_container_width=True) and keywords_input:
            keywords_list = [k.strip() for k in keywords_input.split(',') if k.strip()]
            
            with st.spinner(f"🤖 AI가 '{', '.join(keywords_list)}' 키워드를 심층 분석하는 중..."):
                custom_analysis_result = generate_custom_analysis(
                    df, keywords_list, selected_industry, analysis_type
                )
            
            st.markdown("### 🎯 맞춤 분석 결과")
            st.markdown(custom_analysis_result)
        
        elif not keywords_input and st.button("🎯 맞춤 분석 실행", use_container_width=True):
            st.warning("⚠️ 분석할 키워드를 입력해주세요.")

if __name__ == "__main__":
    main()
